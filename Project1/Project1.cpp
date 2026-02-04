// Project1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.

#include <windows.h>
#include <complex>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>
#include <vector>
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>

double av_omp_3(const double* V, size_t n) {
    unsigned P = omp_get_num_procs();
    unsigned T;
    double* r = static_cast<double*>(calloc(sizeof(double), P));

#pragma omp parallel shared(T)
    {
        unsigned
            t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
        }

        double output = 0.0;
        for (size_t i = t; i < n; i += T)
            output += V[i];
        r[t] = output;
    }

    double sum = 0.0;
    for (size_t i = 0; i < P; i++)
        sum += r[i];

    return sum / n;
}

struct sum_t {
    double v;
    char padding[64 - sizeof(double)];
};

double av_omp_4(const double* V, size_t n) {
    unsigned P = omp_get_num_procs();
    unsigned T;
    struct sum_t* r = static_cast<sum_t*>(calloc(64, P));

#pragma omp parallel shared(T)
    {
        unsigned
            t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
        }

        double output = 0.0;
        for (size_t i = t; i < n; i += T)
            output += V[i];
        r[t].v = output;
    }

    double sum = 0.0;
    for (size_t i = 0; i < P; i++)
        sum += r[i].v;
    free(r);
    return sum / n;
}

void speedtest_avg_openmp(size_t n, size_t exp_count) {
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>> SPEEDTEST OPEN MP AVG " << std::endl;

    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output1.csv"));
    if (!output.is_open())
    {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    output << "T,Time,Avg,Acceleration\n";

    double time_sum_1;
    for (int thread_num = 1; thread_num <= std::thread::hardware_concurrency(); thread_num++) {
        omp_set_num_threads(thread_num);

        double time_sum = 0;
        auto t0 = std::chrono::steady_clock::now();


        for (size_t exp = 0; exp < exp_count; exp++) {

            double* p = static_cast<double*>(malloc(n * sizeof(double)));

            for (size_t i = 0; i < n; i++) {
                p[i] = (double)i;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            av_omp_4(p, n);
            auto t2 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            free(p);
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);

        if (thread_num == 1) {
            time_sum_1 = time_sum;
        }

        std::cout << "AVG: T = " << thread_num << "\t| total  time: " << total_time.count() << "\t| avg time = "
            << time_sum / exp_count << "\tacceleration = " << (time_sum_1 / exp_count) / (time_sum / exp_count) << "\n";

        output << thread_num << "," << total_time.count() << "," << time_sum / exp_count << "," << (time_sum_1 / exp_count) / (time_sum / exp_count) << std::endl;

    }

    output.close();
}

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    speedtest_avg_openmp(1 << 25, 5);
    return 0;
}