#pragma once

#include <iostream>
#include <string>
#include <chrono>

class Timer
{
public:
    Timer(std::string printName);
    void start();
    void endAndPrint();

private:
    std::chrono::steady_clock::time_point m_startTime;
    std::string m_printName;
};