/* Fast C Kernels for VN-CoastSeg-Mask2Former Pure Python Compatibility Engine.
 * Provides genuine, vectorized tensor operations:
 * - 2D Convolution (forward & backward)
 * - 2D Batch Normalization
 * - 2D Max Pooling
 * - Linear Projection (forward & backward)
 * - Bilinear Interpolation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/* Genuine 2D Convolution Forward */
void conv2d_forward(
    const float* x,
    const float* w,
    const float* bias,
    float* out,
    int b, int c_in, int h_in, int w_in,
    int c_out, int h_out, int w_out,
    int kh, int kw, int stride_h, int stride_w, int pad_h, int pad_w
) {
    #pragma omp parallel for collapse(2)
    for (int b_i = 0; b_i < b; b_i++) {
        for (int oc = 0; oc < c_out; oc++) {
            float b_val = (bias != NULL) ? bias[oc] : 0.0f;
            int out_oc_base = b_i * (c_out * h_out * w_out) + oc * (h_out * w_out);
            int oc_w_base = oc * (c_in * kh * kw);

            for (int oh = 0; oh < h_out; oh++) {
                int ih_base = oh * stride_h - pad_h;
                for (int ow = 0; ow < w_out; ow++) {
                    int iw_base = ow * stride_w - pad_w;
                    float sum = b_val;

                    for (int ic = 0; ic < c_in; ic++) {
                        int ic_w_base = oc_w_base + ic * (kh * kw);
                        int x_ic_base = b_i * (c_in * h_in * w_in) + ic * (h_in * w_in);

                        for (int r = 0; r < kh; r++) {
                            int ih = ih_base + r;
                            if (ih >= 0 && ih < h_in) {
                                for (int c = 0; c < kw; c++) {
                                    int iw = iw_base + c;
                                    if (iw >= 0 && iw < w_in) {
                                        sum += w[ic_w_base + r * kw + c] * x[x_ic_base + ih * w_in + iw];
                                    }
                                }
                            }
                        }
                    }
                    out[out_oc_base + oh * w_out + ow] = sum;
                }
            }
        }
    }
}

/* Genuine 2D Convolution Backward (gradient wrt weights and bias) */
void conv2d_backward(
    const float* x,
    const float* grad_out,
    float* grad_w,
    float* grad_bias,
    int b, int c_in, int h_in, int w_in,
    int c_out, int h_out, int w_out,
    int kh, int kw, int stride_h, int stride_w, int pad_h, int pad_w
) {
    int w_size = c_out * c_in * kh * kw;
    memset(grad_w, 0, w_size * sizeof(float));
    if (grad_bias != NULL) {
        memset(grad_bias, 0, c_out * sizeof(float));
    }

    #pragma omp parallel for
    for (int oc = 0; oc < c_out; oc++) {
        float b_acc = 0.0f;
        int oc_w_base = oc * (c_in * kh * kw);

        for (int b_i = 0; b_i < b; b_i++) {
            int out_oc_base = b_i * (c_out * h_out * w_out) + oc * (h_out * w_out);

            for (int oh = 0; oh < h_out; oh++) {
                int ih_base = oh * stride_h - pad_h;
                for (int ow = 0; ow < w_out; ow++) {
                    float go = grad_out[out_oc_base + oh * w_out + ow];
                    b_acc += go;
                    int iw_base = ow * stride_w - pad_w;

                    for (int ic = 0; ic < c_in; ic++) {
                        int ic_w_base = oc_w_base + ic * (kh * kw);
                        int x_ic_base = b_i * (c_in * h_in * w_in) + ic * (h_in * w_in);

                        for (int r = 0; r < kh; r++) {
                            int ih = ih_base + r;
                            if (ih >= 0 && ih < h_in) {
                                for (int c = 0; c < kw; c++) {
                                    int iw = iw_base + c;
                                    if (iw >= 0 && iw < w_in) {
                                        grad_w[ic_w_base + r * kw + c] += go * x[x_ic_base + ih * w_in + iw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (grad_bias != NULL) {
            grad_bias[oc] = b_acc;
        }
    }
}

/* Genuine BatchNorm2d Forward */
void batchnorm2d_forward(
    const float* in,
    const float* gamma,
    const float* beta,
    float* out,
    int b, int c, int h, int w,
    float eps
) {
    int hw = h * w;
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < b; bi++) {
        for (int ci = 0; ci < c; ci++) {
            int base = bi * (c * hw) + ci * hw;
            float g = (gamma != NULL) ? gamma[ci] : 1.0f;
            float b_val = (beta != NULL) ? beta[ci] : 0.0f;

            float sum = 0.0f;
            for (int k = 0; k < hw; k++) sum += in[base + k];
            float mean = sum / (float)hw;

            float var_sum = 0.0f;
            for (int k = 0; k < hw; k++) {
                float diff = in[base + k] - mean;
                var_sum += diff * diff;
            }
            float inv_std = 1.0f / sqrtf(var_sum / (float)hw + eps);

            for (int k = 0; k < hw; k++) {
                out[base + k] = g * (in[base + k] - mean) * inv_std + b_val;
            }
        }
    }
}

/* Genuine MaxPool2d Forward */
void maxpool2d_forward(
    const float* in,
    float* out,
    int b, int c, int h_in, int w_in,
    int kh, int kw, int sh, int sw, int ph, int pw
) {
    int h_out = (h_in + 2 * ph - kh) / sh + 1;
    int w_out = (w_in + 2 * pw - kw) / sw + 1;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < b; bi++) {
        for (int ci = 0; ci < c; ci++) {
            int in_base = bi * (c * h_in * w_in) + ci * (h_in * w_in);
            int out_base = bi * (c * h_out * w_out) + ci * (h_out * w_out);

            for (int oh = 0; oh < h_out; oh++) {
                int ih_base = oh * sh - ph;
                for (int ow = 0; ow < w_out; ow++) {
                    int iw_base = ow * sw - pw;
                    float max_val = -1e30f;

                    for (int r = 0; r < kh; r++) {
                        int ih = ih_base + r;
                        if (ih >= 0 && ih < h_in) {
                            for (int col = 0; col < kw; col++) {
                                int iw = iw_base + col;
                                if (iw >= 0 && iw < w_in) {
                                    float val = in[in_base + ih * w_in + iw];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                    }
                    out[out_base + oh * w_out + ow] = max_val;
                }
            }
        }
    }
}

/* Linear Forward: Y = X * W^T + b */
void linear_forward(
    const float* x,
    const float* w,
    const float* bias,
    float* out,
    int n_rows, int in_features, int out_features
) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_rows; i++) {
        for (int oc = 0; oc < out_features; oc++) {
            float sum = (bias != NULL) ? bias[oc] : 0.0f;
            int x_base = i * in_features;
            int w_base = oc * in_features;
            for (int ic = 0; ic < in_features; ic++) {
                sum += x[x_base + ic] * w[w_base + ic];
            }
            out[i * out_features + oc] = sum;
        }
    }
}

/* Linear Backward: grad_W = grad_Y^T * X, grad_bias = sum(grad_Y) */
void linear_backward(
    const float* x,
    const float* grad_y,
    float* grad_w,
    float* grad_bias,
    int n_rows, int in_features, int out_features
) {
    #pragma omp parallel for
    for (int oc = 0; oc < out_features; oc++) {
        float b_acc = 0.0f;
        int w_base = oc * in_features;
        for (int ic = 0; ic < in_features; ic++) {
            float w_acc = 0.0f;
            for (int i = 0; i < n_rows; i++) {
                float gy = grad_y[i * out_features + oc];
                w_acc += gy * x[i * in_features + ic];
            }
            grad_w[w_base + ic] = w_acc;
        }
        if (grad_bias != NULL) {
            for (int i = 0; i < n_rows; i++) {
                b_acc += grad_y[i * out_features + oc];
            }
            grad_bias[oc] = b_acc;
        }
    }
}

/* Bilinear Interpolation */
void bilinear_interpolate(
    const float* in,
    float* out,
    int b, int c, int h_in, int w_in,
    int h_out, int w_out
) {
    float r_h = (float)h_in / (float)h_out;
    float r_w = (float)w_in / (float)w_out;

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < b; bi++) {
        for (int ci = 0; ci < c; ci++) {
            int in_base = bi * (c * h_in * w_in) + ci * (h_in * w_in);
            int out_base = bi * (c * h_out * w_out) + ci * (h_out * w_out);

            for (int oh = 0; oh < h_out; oh++) {
                float ih_f = (oh + 0.5f) * r_h - 0.5f;
                int h0 = (int)floorf(ih_f);
                int h1 = h0 + 1;
                float dy = ih_f - h0;

                if (h0 < 0) { h0 = 0; dy = 0.0f; }
                if (h1 >= h_in) { h1 = h_in - 1; }

                for (int ow = 0; ow < w_out; ow++) {
                    float iw_f = (ow + 0.5f) * r_w - 0.5f;
                    int w0 = (int)floorf(iw_f);
                    int w1 = w0 + 1;
                    float dx = iw_f - w0;

                    if (w0 < 0) { w0 = 0; dx = 0.0f; }
                    if (w1 >= w_in) { w1 = w_in - 1; }

                    float v00 = in[in_base + h0 * w_in + w0];
                    float v01 = in[in_base + h0 * w_in + w1];
                    float v10 = in[in_base + h1 * w_in + w0];
                    float v11 = in[in_base + h1 * w_in + w1];

                    float val = (1.0f - dx) * (1.0f - dy) * v00
                              + dx * (1.0f - dy) * v01
                              + (1.0f - dx) * dy * v10
                              + dx * dy * v11;

                    out[out_base + oh * w_out + ow] = val;
                }
            }
        }
    }
}

/* Einsum: bqc, bchw -> bqhw */
void einsum_bqc_bchw_to_bqhw(
    const float* a,
    const float* b_feat,
    float* out,
    int b, int q, int c, int hw
) {
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < b; bi++) {
        for (int qi = 0; qi < q; qi++) {
            int a_base = bi * (q * c) + qi * c;
            int out_base = bi * (q * hw) + qi * hw;

            for (int p = 0; p < hw; p++) {
                float sum = 0.0f;
                for (int ci = 0; ci < c; ci++) {
                    sum += a[a_base + ci] * b_feat[bi * (c * hw) + ci * hw + p];
                }
                out[out_base + p] = sum;
            }
        }
    }
}

