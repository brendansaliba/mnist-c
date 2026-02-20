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
matrix* mat_load(mem_arena* arena, uint32_t rows, uint32_t cols, const char* filename);
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
bool mat_mul(matrix* out, const matrix* a, const matrix* b, bool zero_out, bool transpose_a, bool transpose_b);

bool mat_relu(matrix* out, const matrix* in);
bool mat_softmax(matrix* out, const matrix* in);
bool mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
bool mat_relu_add_grad(matrix* out, const matrix* in);
bool mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
bool mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);

void draw_mnist_digit(float* data);

int main() {
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    matrix* train_images = mat_load(perm_arena, 60000, 784, "dataset/train_images.mat");
    matrix* test_images = mat_load(perm_arena, 10000, 784, "dataset/test_images.mat");
    matrix* train_labels = mat_create(perm_arena, 60000, 10);
    matrix* test_labels = mat_create(perm_arena, 10000, 10);

    {
        matrix* train_labels_file = mat_load(perm_arena, 60000, 1, "dataset/train_labels.mat");
        matrix* test_labels_file = mat_load(perm_arena, 10000, 1, "dataset/test_labels.mat");

        for (uint32_t i = 0; i < 60000; i++) {
            uint32_t num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f;
        }

        for (uint32_t i = 0; i < 10000; i++) {
            uint32_t num = test_labels_file->data[i];
            test_labels->data[i * 10 + num] = 1.0f;
        }
    }

    draw_mnist_digit(train_images->data);
    for (uint32_t i = 0; i < 10; i++) {
        printf("%.0f ", train_labels->data[i]);
    }
    printf("\n");
    
    arena_destroy(perm_arena);
    
    return 0;
}

void draw_mnist_digit(float* data) {
    for (uint32_t y = 0; y < 28; y++) {
        for (uint32_t x = 0; x < 28; x++) {
            float num = data[ x + y * 28];
            uint32_t col = 232 + (uint32_t)(num * 24);
            printf("\x1b[48;5;%dm ", col);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

matrix* mat_create(mem_arena* arena, uint32_t rows, uint32_t cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);
    
    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, float, (uint64_t)rows * cols);
    
    return mat;
}

matrix* mat_load(mem_arena* arena, uint32_t rows, uint32_t cols, const char* filename) {
    matrix* mat = mat_create(arena, rows, cols);

    FILE* f = fopen(filename, "rb");

    fseek(f, 0, SEEK_END);
    uint64_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size = MIN(size, sizeof(float)* rows * cols);

    fread(mat->data, 1, size, f);

    fclose(f);

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

bool mat_mul(matrix* out, const matrix* a, const matrix* b, bool zero_out, bool transpose_a, bool transpose_b) {
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

bool mat_relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    uint64_t size = (uint64_t)out->rows * out->cols;
    for (uint64_t i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
    }

    return true;
}

bool mat_softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }

    uint64_t size = (uint64_t)out->rows * out->cols;
    float sum = 0.0f;
    for (uint64_t i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }

    mat_scale(out, 1.0f/sum);

    return true;
}

bool mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) { return false; }

    uint64_t size = (uint64_t)out->rows * out->cols;
    for (uint64_t i = 0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i]);
    }

    return true;
}

bool mat_relu_add_grad(matrix* out, const matrix* in);
bool mat_softmax_add_grad(matrix* out, const matrix* softmax_out);
bool mat_cross_entropy_add_grad(matrix* out, const matrix* p, const matrix* q);