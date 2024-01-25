#include <iostream>
#include <vector>

class LowPassFilter
{
private:
    double alpha;
    double lastOutput;

public:
    LowPassFilter(double alpha) : alpha(alpha), lastOutput(0) {}

    double process(double input)
    {
        double output = alpha * input + (1 - alpha) * lastOutput;
        lastOutput = output;
        return output;
    }
};

int main()
{
    // 示例：使用0.1作为alpha值
    LowPassFilter lpf(0.1);

    // 假设的输入数据
    std::vector<double> inputData = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0};

    std::cout << "Filtered data: ";
    for (double data : inputData)
    {
        std::cout << lpf.process(data) << " ";
    }

    return 0;
}
