#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

std::vector<std::vector<int>> read_matrix_from_file(const std::string& filePath) {
    std::ifstream inputFile(filePath);

    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
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

pair<int, vector<vector<pair<int, int>>>> calculate_makespan(vector<int>& order, vector<vector<int>>& processing_times) {
    vector<vector<int>> ordered_processing_times(order.size());
    for (int i = 0; i < order.size(); i++) {
        ordered_processing_times[i] = processing_times[order[i]];
    }

    int machines = processing_times.size();
    int jobs = processing_times[0].size();

    vector<vector<pair<int, int>>> structure(machines, vector<pair<int, int>>(jobs));

    structure[0][0] = make_pair(0, ordered_processing_times[0][0]);

    for (int i = 1; i < jobs; ++i) {
        structure[0][i] = make_pair(structure[0][i - 1].second, structure[0][i - 1].second + ordered_processing_times[0][i]);
    }

    for (int i = 1; i < machines; ++i) {
        structure[i][0] = make_pair(structure[i - 1][0].second, structure[i - 1][0].second + ordered_processing_times[i][0]);
    }

    for (int i = 1; i < machines; ++i) {
        for (int j = 1; j < jobs; ++j) {
            int start = max(structure[i - 1][j].second, structure[i][j - 1].second);
            structure[i][j] = make_pair(start, start + ordered_processing_times[i][j]);
        }
    }
    return make_pair(structure[machines - 1][jobs - 1].second, structure);
}

double probability(const vector<double>& pheromones, double heuristic, int current_point,
    int considerated_point, const vector<int>& unvisited, double a, double b, int n) {
    double numerator = pow(pheromones[current_point * n + considerated_point], a) * pow(heuristic, b);
    double denominator = 0.0;

    for (int l = 0; l < unvisited.size(); ++l) {
        denominator += pow(pheromones[current_point * n + l], a) * pow(heuristic, b);
    }

    return numerator / denominator;
}

pair<vector<int>, int> ant_colony_optimization_with_2opt(vector<vector<int>>& processing_times,
    int iterations_count, int ants_count,
    double evaporation_rate, double a, double b, int rank, int size) {
    int n = processing_times.size();
    int num_threads = 4;
    vector<double> pheromones(n * n);
    fill(pheromones.begin(), pheromones.end(), 1);

    vector<int> best_order(n);
    int best_order_makespan = numeric_limits<int>::max();

    for (int iteration = 0; iteration < iterations_count; ++iteration) {
        vector<int> orders;
        vector<int> order_makespans;

        int ants_per_process = ants_count / size;
        int start_ant = rank * ants_per_process;
        int end_ant = (rank == size - 1) ? ants_count : start_ant + ants_per_process;

        for (int ant = start_ant; ant < end_ant; ++ant) {
            int current_job = rand() % n;
            vector<bool> ordered(n, false);
            ordered[current_job] = true;
            vector<int> order = { current_job };
            int order_makespan = 0;

            while (find(ordered.begin(), ordered.end(), false) != ordered.end()) {
                vector<int> unordered;
                for (int i = 0; i < n; ++i) {
                    if (!ordered[i]) {
                        unordered.push_back(i);
                    }
                }

                vector<double> probabilities(unordered.size());
                for (int i = 0; i < unordered.size(); ++i) {
                    double mean = 0;
                    for (int j = 0; j < processing_times[0].size(); ++j) {
                        mean += processing_times[unordered[i]][j];
                    }
                    mean /= processing_times[0].size();
                    probabilities[i] = probability(pheromones, 1.0 / mean,
                        current_job, unordered[i], unordered, a, b, n);
                }

                int next_job = unordered[rand() % unordered.size()];
                order.push_back(next_job);
                ordered[next_job] = true;
                current_job = next_job;
            }

            order_makespan = calculate_makespan(order, processing_times).first;
            for (int i = 0; i < n; i++) {
                orders.push_back(order[i]);
            }
            order_makespans.push_back(order_makespan);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        vector<int> all_orders;
        vector<int> all_order_makespans;

        if (rank == 0) {
            all_orders.resize(ants_count * n);
            all_order_makespans.resize(ants_count);
        }

        MPI_Gather(orders.data(), ants_per_process * n, MPI_INT, all_orders.data(), ants_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(order_makespans.data(), ants_per_process, MPI_INT, all_order_makespans.data(), ants_per_process, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    pheromones[i * n + j] = (1 - evaporation_rate);
                }
            }
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < ants_count; ++i) {
                double delta = 1 / all_order_makespans[i];
                for (int j = 0; j < (n - 1); ++j) {
                    pheromones[all_orders[n * i + j] * n + all_orders[n * i + (j + 1)]] += delta;
                }
                pheromones[all_orders[n * i + (n - 1)] * n + all_orders[n * i]] += delta;
            }

            int best_order_index = 0;
            for (int i = 0; i < ants_count; ++i) {
                if (all_order_makespans[i] < best_order_makespan) {
                    best_order_makespan = all_order_makespans[i];
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

    return make_pair(best_order, best_order_makespan);
}

vector<int> brute_force(vector<vector<int>>& processing_times) {
    int n = processing_times.size();
    vector<int> jobs(n);
    iota(jobs.begin(), jobs.end(), 0);

    vector<int> best_order;
    int best_order_makespan = numeric_limits<int>::max();

    do {
        auto current_makespan = calculate_makespan(jobs, processing_times);
        if (current_makespan.first < best_order_makespan) {
            best_order_makespan = current_makespan.first;
            best_order = jobs;
        }
    } while (next_permutation(jobs.begin(), jobs.end()));

    return best_order;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(rank);

    vector<vector<int>> processing_times = read_matrix_from_file("fssp_100_10.txt");

    auto start_time = std::chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);

    auto result = ant_colony_optimization_with_2opt(processing_times, 100, 24, 0.1, 4.0, 10.0, rank, size);

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (rank == 0) {
        std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
        cout << "Best Order Makespan: " << result.second << endl;
        cout << "Best Order: ";
        for (int value : result.first) {
            cout << value << " ";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}