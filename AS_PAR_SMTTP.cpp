#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <numeric>
#include <mpi.h>
#include <omp.h>
#include <chrono>

double job_delay_smttp(std::vector<int>& order, const std::vector<double>& processing_times, double due_date) {
    double processing_time = 0.0;

    for (int i = 0; i < order.size(); ++i) {
        processing_time += processing_times[order[i]];
    }

    double obj = processing_time - due_date;
    return std::max(obj, 0.0);
}

double calculate_order_delay_smttp(std::vector<int>& order, const std::vector<double>& processing_times, const std::vector<double>& due_dates) {
    double order_delay = 0;

    for (int i = 0; i < order.size(); ++i) {
        std::vector<int> order_part(order.begin(), order.begin() + i + 1);
        order_delay += job_delay_smttp(order_part, processing_times, due_dates[order[i]]);
    }

    return order_delay;
}

double probability_smttp(const std::vector<double>& pheromones, double heuristic, int current_point,
    int considerated_point, const std::vector<int>& unvisited, double a, double b, int n) {
    double numerator = std::pow(pheromones[current_point * n + considerated_point], a) * std::pow(heuristic, b);
    double denominator = 0.0;

    for (int l = 0; l < unvisited.size(); ++l) {
        denominator += std::pow(pheromones[current_point * n + l], a) * std::pow(heuristic, b);
    }

    return numerator / denominator;
}

double brute_force_smttp(const std::vector<double>& processing_times, const std::vector<double>& due_dates) {
    int n = processing_times.size();
    std::vector<int> jobs(n);
    std::iota(jobs.begin(), jobs.end(), 0);

    std::vector<int> best_order;
    double best_order_delay = std::numeric_limits<double>::infinity();

    do {
        double current_delay = calculate_order_delay_smttp(jobs, processing_times, due_dates);
        if (current_delay < best_order_delay) {
            best_order_delay = current_delay;
            best_order = jobs;
        }
    } while (std::next_permutation(jobs.begin(), jobs.end()));

    return best_order_delay;
}

std::pair<std::vector<int>, double> ant_colony_optimization_smttp(std::vector<double>& processing_times, std::vector<double>& due_dates,
    int iterations_count, int ants_count, double evaporation_rate, double a, double b, int rank, int size) {
    int n = processing_times.size();
    int num_threads = 4;

    std::vector<double> pheromones(n * n);
    std::fill(pheromones.begin(), pheromones.end(), 1);

    std::vector<int> best_order(n);
    double best_order_delay = std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < iterations_count; ++iteration) {
        std::vector<int> orders;
        std::vector<double> order_delays;

        int ants_per_process = ants_count / size;
        int start_ant = rank * ants_per_process;
        int end_ant = (rank == size - 1) ? ants_count : start_ant + ants_per_process;

        for (int ant = start_ant; ant < end_ant; ++ant) {
            int current_job = rand() % n;
            std::vector<bool> ordered(n, false);
            ordered[current_job] = true;
            std::vector<int> order = { current_job };
            double order_delay = 0;

            while (std::find(ordered.begin(), ordered.end(), false) != ordered.end()) {
                std::vector<int> unordered;
                for (int i = 0; i < n; ++i) {
                    if (!ordered[i]) {
                        unordered.push_back(i);
                    }
                }

                std::vector<double> probabilities(unordered.size());
                for (int i = 0; i < unordered.size(); ++i) {
                    probabilities[i] = probability_smttp(pheromones, 1.0 / processing_times[unordered[i]],
                        current_job, unordered[i], unordered, a, b, n);
                }

                int next_job = unordered[rand() % unordered.size()];
                order.push_back(next_job);
                order_delay += job_delay_smttp(order, processing_times, due_dates[next_job]);
                ordered[next_job] = true;
                current_job = next_job;
            }

            order_delay = calculate_order_delay_smttp(order, processing_times, due_dates);
            for (int i = 0; i < n; i++) {
                orders.push_back(order[i]);
            }
            order_delays.push_back(order_delay);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> all_orders;
        std::vector<double> all_order_delays;

        if (rank == 0) {
            all_orders.resize(ants_count * n);
            all_order_delays.resize(ants_count);
        }

        MPI_Gather(orders.data(), ants_per_process * n, MPI_INT, all_orders.data(), ants_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(order_delays.data(), ants_per_process, MPI_DOUBLE, all_order_delays.data(), ants_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    pheromones[i * n + j] *= (1 - evaporation_rate);
                }
            }
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ants_count; ++i) {
                double delta = 1 / all_order_delays[i];
                for (int j = 0; j < (n - 1); ++j) {
                    pheromones[all_orders[n * i + j] * n + all_orders[n * i + (j + 1)]] += delta;
                }
                pheromones[all_orders[n * i + (n - 1)] * n + all_orders[n * i]] += delta;
            }

            int best_order_index = 0;
            for (int i = 0; i < ants_count; ++i) {
                if (all_order_delays[i] < best_order_delay) {
                    best_order_delay = all_order_delays[i];
                    best_order_index = i;
                }
            }

            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; i++) {
                best_order[i] = all_orders[n * best_order_index + i];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(pheromones.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    return std::make_pair(best_order, best_order_delay);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> processing_times = { 2, 3, 5 , 8, 10, 26, 6 , 65 , 78 };
    std::vector<double> due_dates = { 10, 76, 6 , 15, 15, 65, 56 , 89, 78 };

    MPI_Barrier(MPI_COMM_WORLD);

    std::srand(rank);
    auto start_time = std::chrono::high_resolution_clock::now();

    std::pair<std::vector<int>, double> result = ant_colony_optimization_smttp(processing_times, due_dates, 10, 4, 0.8, 5, 1, rank, size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (rank == 0) {
        std::cout << result.second << std::endl;
        std::cout << duration.count() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
