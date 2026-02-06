#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>  // for std::setprecision

class CsvWriter {
public:
    CsvWriter(const std::string& filename, int precision = 6)
        : file(filename), floatPrecision(precision)
    {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }
        file << std::fixed << std::setprecision(floatPrecision);
    }

    // Writes header to the CSV file
    void writeHeader(const std::vector<std::string>& header) {
        for (size_t i = 0; i < header.size(); ++i) {
            file << header[i];
            if (i < header.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    // Writes one row to the CSV file
    void writeRow(const std::vector<float>& row) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    // Check if file is ready
    bool good() const {
        return file.good();
    }

    // Flush contents if needed
    void flush() {
        file.flush();
    }

    ~CsvWriter() {
        file.close();
    }

private:
    std::ofstream file;
    int floatPrecision;
};
