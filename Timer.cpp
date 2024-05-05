#include "Timer.hpp"

Timer::Timer(std::string printName)
{
    m_printName = printName;
}

void Timer::start()
{
    m_startTime = std::chrono::high_resolution_clock::now();
    std::cout << m_printName << " execution started\n";
}

void Timer::endAndPrint()
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_startTime).count();
    std::cout << m_printName << " time taken: " << duration << " microseconds\n";
}
