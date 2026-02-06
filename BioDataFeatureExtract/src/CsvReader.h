#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CsvReader {
public:
    CsvReader(const std::string& filename)
        : file(filename)
    {
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }

    // Reads the next row into a vector of floats
    bool nextRow(std::vector<int>& row) {
        row.clear();
        std::string line;
        if (!std::getline(file, line)) return false;

        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stoi(value));
            } catch (const std::exception& e) {
                std::cerr << "Invalid float value: " << value << "\n";
                row.push_back(0.0f);  // fallback or skip as you prefer
            }
        }
        return true;
    }

    // Check if the file is still good
    bool good() const {
        return file.good();
    }

private:
    std::ifstream file;
};
