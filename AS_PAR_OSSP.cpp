#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <sstream>

typedef std::tuple<int, int, int> StructureTuple;

std::vector<std::vector<int>> read_matrix_from_file(const std::string& filePath) {
    std::ifstream inputFile(filePath);

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        // You might want to handle this error differently depending on your application
        exit(1);
    }

    std::vector<std::vector<int>> matrix;

    std::string line;
    while (std::getline(inputFile, line)) {
        std::vector<int> row;
        std::istringstream iss(line);
        int num;
        while (iss >> num) {
            row.push_back(num);
        }
        matrix.push_back(row);
    }

    inputFile.close();

    return matrix;
}

std::tuple<int, int, int> transform_from_representating_value_to_processing_times_value(int representating_value, const std::vector<std::vector<int>>& processing_times) {
    int machines_quantity = processing_times.size();
    int jobs_quantity = processing_times[0].size();

    int quotient = representating_value / jobs_quantity;
    int remainder = representating_value % jobs_quantity;
    int machine = quotient;
    int operation = remainder;

    return std::make_tuple(machine, operation, processing_times[machine][operation]);
}

std::pair<int, std::vector<std::vector<StructureTuple>>> calculate_makespan_ossp(const std::vector<int>& order, const std::vector<std::vector<int>>& processing_times) {
    int machines_quantity = processing_times.size();
    int jobs_quantity = processing_times[0].size();

    std::vector<int> max_ends_of_operation_on_machines(machines_quantity, 0);
    std::vector<int> max_ends_of_jobs(jobs_quantity, 0);
    std::vector<int> assigned_operations_quantity_on_machines(machines_quantity, 0);

    std::vector<std::vector<StructureTuple>> structure(machines_quantity, std::vector<StructureTuple>(jobs_quantity, std::make_tuple(0, 0, 0)));

    for (int i = 0; i < order.size(); ++i) {
        std::tuple<int, int, int> processing_time_value = transform_from_representating_value_to_processing_times_value(order[i], processing_times);
        int start = std::max(max_ends_of_operation_on_machines[std::get<0>(processing_time_value)], max_ends_of_jobs[std::get<1>(processing_time_value)]);
        int end = start + std::get<2>(processing_time_value);
        max_ends_of_operation_on_machines[std::get<0>(processing_time_value)] = end;
        max_ends_of_jobs[std::get<1>(processing_time_value)] = end;
        assigned_operations_quantity_on_machines[std::get<0>(processing_time_value)] += 1;
        structure[std::get<0>(processing_time_value)][assigned_operations_quantity_on_machines[std::get<0>(processing_time_value)] - 1] = std::make_tuple(start, end, order[i]);
    }

    int makespan = 0;
    for (const auto& row : structure) {
        for (const StructureTuple& tuple : row) {
            makespan = std::max(makespan, std::get<1>(tuple));
        }
    }

    return std::make_pair(makespan, structure);
}

double get_heuristic_value(const std::vector<std::vector<int>>& processing_times, int representating_value) {
    int processing_time_value = std::get<2>(transform_from_representating_value_to_processing_times_value(representating_value, processing_times));
    return 1.0 / processing_time_value;
}

double probability_ossp(const std::vector<double>& pheromones, double heuristic, int current_point,
    int considerated_point, const std::vector<int>& unvisited, double a, double b, int n) {
    double numerator = std::pow(pheromones[current_point * n + considerated_point], a) * std::pow(heuristic, b);
    double denominator = 0.0;

    for (int l = 0; l < unvisited.size(); ++l) {
        denominator += std::pow(pheromones[current_point * n + l], a) * std::pow(heuristic, b);
    }

    return numerator / denominator;
}

std::pair<std::vector<int>, int> ant_colony_optimization_ossp(const std::vector<std::vector<int>>& processing_times, int iterations_count, int ants_count,
    double evaporation_rate, double a, double b, int rank, int size) {
    int machines_quantity = processing_times.size();
    int jobs_quantity = processing_times[0].size();
    int k = machines_quantity * jobs_quantity;
    int num_threads = 4;

    std::vector<double> pheromones(k * k);
    std::fill(pheromones.begin(), pheromones.end(), 1);

    std::vector<int> best_order(k);
    int best_order_makespan = INT_MAX;

    for (int iteration = 0; iteration < iterations_count; ++iteration) {
        std::vector<int> orders;
        std::vector<int> order_makespans;

        int ants_per_process = ants_count / size;

        int start_ant = rank * ants_per_process;
        int end_ant = (rank == size - 1) ? ants_count : start_ant + ants_per_process;


        for (int ant = start_ant; ant < end_ant; ++ant) {
            int current_job = rand() % k;
            std::vector<bool> ordered(k, false);
            ordered[current_job] = true;
            std::vector<int> order = { current_job };

            while (find(ordered.begin(), ordered.end(), false) != ordered.end()) {
                std::vector<int> unordered;
                for (int i = 0; i < k; ++i) {
                    if (!ordered[i]) {
                        unordered.push_back(i);
                    }
                }

                std::vector<double> probabilities(unordered.size(), 0.0);
                for (int i = 0; i < unordered.size(); ++i) {
                    double heuristic = get_heuristic_value(processing_times, unordered[i]);
                    probabilities[i] = probability_ossp(pheromones, heuristic, current_job, unordered[i], unordered, a, b, k);
                }

                int next_job = unordered[rand() % unordered.size()];
                order.push_back(next_job);
                ordered[next_job] = true;
                current_job = next_job;
            }

            for (int i = 0; i < k; i++) {
                orders.push_back(order[i]);
            }
            int order_makespan = calculate_makespan_ossp(order, processing_times).first;
            order_makespans.push_back(order_makespan);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> all_orders;
        std::vector<int> all_order_makespans;

        if (rank == 0) {
            all_orders.resize(ants_count * k);
            all_order_makespans.resize(ants_count);
        }

        MPI_Gather(orders.data(), ants_per_process * k, MPI_INT, all_orders.data(), ants_per_process * k, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(order_makespans.data(), ants_per_process, MPI_INT, all_order_makespans.data(), ants_per_process, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < k; ++i) {
                for (int j = 0; j < k; ++j) {
                    pheromones[i * k + j] = (1 - evaporation_rate);
                }
            }

            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ants_count; ++i) {
                double delta = 1 / all_order_makespans[i];
                for (int j = 0; j < (k - 1); ++j) {
                    pheromones[all_orders[k * i + j] * k + all_orders[k * i + (j + 1)]] += delta;
                }
                pheromones[all_orders[k * i + (k - 1)] * k + all_orders[k * i]] += delta;
            }

            int best_order_index = 0;
            for (int i = 0; i < ants_count; ++i) {
                if (all_order_makespans[i] < best_order_makespan) {
                    best_order_makespan = all_order_makespans[i];
                    best_order_index = i;
                }
            }

            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < k; i++) {
                best_order[i] = all_orders[k * best_order_index + i];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(pheromones.data(), k * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    return std::make_pair(best_order, best_order_makespan);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::srand(rank);

    std::vector<std::vector<int>> processing_times = read_matrix_from_file("ossp_10_10.txt");
    auto start_time = std::chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);

    std::pair<std::vector<int>, int> result = ant_colony_optimization_ossp(processing_times, 50, 12, 0.8, 10.0, 1.0, rank, size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (rank == 0) {
        std::cout << duration.count() << std::endl;
        std::cout << result.second << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
