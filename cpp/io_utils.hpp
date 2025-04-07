#pragma once
#include <vector>
#include <complex>
#include <fstream>
#include <string>

inline std::vector<std::complex<float>> read_complex_data(const std::string &path)
{
    std::ifstream fin(path);
    float re, im;
    std::vector<std::complex<float>> data;
    while (fin >> re >> im)
    {
        data.emplace_back(re, im);
    }
    return data;
}

inline void write_complex_data(const std::vector<std::complex<float>> &data, const std::string &path)
{
    std::ofstream fout(path);
    for (const auto &c : data)
    {
        fout << c.real() << " " << c.imag() << "\n";
    }
}