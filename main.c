#include "base.h"
#include "arena.h"
#include "prng.h"

#include "arena.c"
#include "prng.c"

typedef struct {
    uint32_t rows, cols;
    float* data;
} matrix;

matrix* mat_create(mem_arena* arena, uint32_t rows, uint32_t cols);
bool mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, float x);
float mat_sum(matrix* mat);
bool mat_add(matrix* out, const matrix* a, const matrix* b);
void mat_scale(matrix* mat, float scale);
bool mat_sub(matrix* out, const matrix* a, const matrix* b);

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b);
void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b);
void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b);
void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b);
bool mat_mul(matrix* out, const matrix* a, const, matrix* b, bool zero_out, bool transpose_a, bool transpose_b);

bool mat_relu(matrix* out, const matrix* in);
bool mat_softmax(matrix* out, const matrix* in);
bool mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
bool mat_relu_add_grad(matrix* out, const matrix* in);
bool mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
bool mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);


int main() {
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));
    
    arena_destroy(perm_arena);
    
    return 0;
}


matrix* mat_create(mem_arena* arena, uint32_t rows, uint32_t cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);
    
    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, float, (uint64_t)rows * cols);
    
    return mat;
}

bool mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        return false;
    }
    
    memcpy(dst->data, src->data, sizeof(float) * (uint64_t)dst->rows * dst->cols);
    
    return true;
}

void mat_clear(matrix* mat) {
    memset(mat->data, 0, sizeof(float) * (uint64_t)mat->rows * mat->cols);
}

void mat_fill(matrix* mat, float x) {
    uint64_t size = (uint64_t)mat->rows * mat->cols;

    for (uint64_t i = 0; i < size; i++) {
        mat->data[i] = x;
    }
}

void mat_scale(matrix* mat, float scale) {
    uint64_t size = (uint64_t)mat->rows * mat->cols;
    
    for (uint64_t i = 0; i < size; i++) {
        mat->data[i] = scale;
    }
    
}

float mat_sum(matrix* mat) {
    uint64_t size = (uint64_t)mat->rows * mat->cols;
    float sum = 0.0f;

    for (uint64_t i = 0; i < size; i++) {
        sum += mat->data[i];
    }
    
    return sum;
}

bool mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    uint64_t size = (uint64_t)out->rows * out->cols;
    for (uint64_t i =  0; i < size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return true;
}

bool mat_sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    uint64_t size = (uint64_t)out->rows * out->cols;
    for (uint64_t i =  0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return true;
}

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (uint64_t i = 0; i < out->rows; i++) {
        for (uint64_t k = 0; k < a->cols; k++) {
            for (uint64_t j = 0; j < a->cols; j++) {
                out->data[j + i * a->cols] += a->data[k + i * a->cols] * b->data[j + k * b->cols];
            }

        }
    }
}
void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (uint64_t i = 0; i < out->rows; i++) {
        for (uint64_t j = 0; j < out->cols; j++) {
            for (uint64_t k = 0; k < a->cols; k++) {
                out->data[j + i * a->cols] += a->data[k + i * a->cols] * b->data[k + j * b->cols];
            }

        }
    }

}
void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (uint64_t k = 0; k < a->rows; k++) {
        for (uint64_t i = 0; i < out->rows; i++) {
            for (uint64_t j = 0; j < out->cols; j++) {
                out->data[j + i * a->cols] += a->data[i + k * a->cols] * b->data[j + k * b->cols];
            }

        }
    }

}
void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (uint64_t i = 0; i < out->rows; i++) {
        for (uint64_t j = 0; j < out->cols; j++) {
                for (uint64_t k = 0; k < a->cols; k++) {
                out->data[j + i * a->cols] += a->data[i + k * a->cols] * b->data[k + j * b->cols];
            }

        }
    }

}

bool mat_mul(matrix* out, const matrix* a, const, matrix* b, bool zero_out, bool transpose_a, bool transpose_b) {
    uint32_t a_rows = transpose_a ? a->cols : a->rows;
    uint32_t a_cols = transpose_a ? a->rows : a->cols;
    uint32_t b_rows = transpose_b ? b->cols : b->rows;
    uint32_t b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows) { return false; }
    if (out->rows != a->rows || out->cols != b_cols) { return false; }

    if (zero_out) {
        mat_clear(out);
    }

    uint32_t transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0b00: { _mat_mul_nn(out, a, b); } break;
        case 0b01: { _mat_mul_nt(out, a, b); } break;
        case 0b10: { _mat_mul_tn(out, a, b); } break;
        case 0b11: { _mat_mul_tt(out, a, b); } break;
    }

    return true;
} 

bool mat_relu(matrix* out, const matrix* in);
bool mat_softmax(matrix* out, const matrix* in);
bool mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
bool mat_relu_add_grad(matrix* out, const matrix* in);
bool mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
bool mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);