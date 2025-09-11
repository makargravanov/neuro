export module Layer;

export import Neuron;
import std;
import types;

export import constants;

export class Layer {
    std::vector<Neuron> _neurons;
public:

    Layer() = default;

    explicit Layer(const std::vector<Weights>& weights) {
        _neurons.resize(weights.size());
        for (u32 i = 0; i < weights.size(); i++) {
            _neurons.at(i) = Neuron(weights.at(i),-1.5);
        }
    };

    explicit Layer(const i32 numberOfNeurons, i32 lastNumberOfNeurons) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution dis(-0.5f, 0.5f);

        _neurons.resize(numberOfNeurons);
        for (u32 i = 0; i < numberOfNeurons; i++) {
            Weights initialWeights(lastNumberOfNeurons);
            for(f32& weight : initialWeights) {
                weight = dis(gen); // Присваиваем случайный вес
            }
            // Смещение (bias) тоже можно инициализировать случайно,
            // или оставить 0, или небольшое значение.
            f32 initialBias = dis(gen);
            _neurons.at(i) = Neuron(initialWeights, initialBias);
        }
    };

    std::vector<Neuron>& getNeurons() { return _neurons; }
    const std::vector<Neuron>& getNeurons() const { return _neurons; }

    Output activate(const Input& input) {
        Output output;
        output.reserve(_neurons.size());
        for (auto& neuron : _neurons) {
            output.push_back(neuron.activation(input));
        }

        return output;
    }
};
