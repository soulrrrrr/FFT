#pragma once
#include <vector>

// now only support radix 2, 4, 8
void plan_fft(std::vector<int> &plan, size_t n)
{
    while (n > 1)
    {
        // if (n % 8 == 0)
        // {
        //     plan.push_back(8);
        //     n /= 8;
        // }
        // else
        if (n % 4 == 0)
        {
            plan.push_back(4);
            n /= 4;
        }
        else if (n % 2 == 0)
        {
            plan.push_back(2);
            n /= 2;
        }
        else
        {
            break;
        }
    }
}