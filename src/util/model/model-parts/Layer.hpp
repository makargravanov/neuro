
#ifndef LAYER_HPP
#define LAYER_HPP

#include "ComputePolicies.h"
#include "../../types/eigen_types.hpp"
#include "ocl-core/OclBuffer.hpp"

template<typename ActivationPolicy, typename ComputePolicy>
class Layer {
public:
    using ActivationPolicyT = ActivationPolicy;

private:
    std::variant<WeightMatrix, OclBuffer> _weights;
    std::variant<BiasVector, OclBuffer> _biases;

    std::variant<Input, OclBuffer> _lastInput;
    std::variant<Output, OclBuffer> _lastOutput;

    u32 _numberOfNeurons;
    u32 _lastNumberOfNeurons;

public:
    Layer() = default;

    explicit Layer(const u32 numberOfNeurons, u32 lastNumberOfNeurons)
        : _numberOfNeurons(numberOfNeurons), _lastNumberOfNeurons(lastNumberOfNeurons)
    {
        if constexpr (std::is_same_v<ComputePolicy, CpuEigenPolicy>) {
            _weights = WeightMatrix::Random(numberOfNeurons, lastNumberOfNeurons);
            _biases = BiasVector::Random(numberOfNeurons);
        } else {
            WeightMatrix tempWeights = WeightMatrix::Random(numberOfNeurons, lastNumberOfNeurons);
            BiasVector tempBiases = BiasVector::Random(numberOfNeurons);

            OclBuffer weightsBuffer, biasesBuffer;
            weightsBuffer.create(tempWeights.size() * sizeof(f32));
            biasesBuffer.create(tempBiases.size() * sizeof(f32));
            weightsBuffer.write(tempWeights.data());
            biasesBuffer.write(tempBiases.data());

            _weights = std::move(weightsBuffer);
            _biases = std::move(biasesBuffer);
        }
    };

    void activate_inplace(std::variant<Input, OclBuffer>&& input, u32 batchSize) {
        _lastInput = std::move(input);

        if constexpr (std::is_same_v<ComputePolicy, CpuEigenPolicy>) {
            auto& input_cpu = std::get<Input>(_lastInput);
            auto& weights_cpu = std::get<WeightMatrix>(_weights);
            auto& biases_cpu = std::get<BiasVector>(_biases);
            Input z = ComputePolicy::forwardPass(weights_cpu, input_cpu, biases_cpu);
            _lastOutput = ComputePolicy::template activate<ActivationPolicy>(z);
        } else {
            auto& input_gpu = std::get<OclBuffer>(_lastInput);
            auto& weights_gpu = std::get<OclBuffer>(_weights);
            auto& biases_gpu = std::get<OclBuffer>(_biases);
            _lastOutput = ComputePolicy::template activate<ActivationPolicy>(
                input_gpu, weights_gpu, biases_gpu,
                _numberOfNeurons, _lastNumberOfNeurons, batchSize
            );
        }
    }

    WeightMatrix& getWeights_cpu() { return std::get<WeightMatrix>(_weights); }
    BiasVector& getBiases_cpu() { return std::get<BiasVector>(_biases); }
    OclBuffer& getWeights_gpu() { return std::get<OclBuffer>(_weights); }
    OclBuffer& getBiases_gpu() { return std::get<OclBuffer>(_biases); }

    [[nodiscard]] const auto& getLastOutput() const { return _lastOutput; }
    [[nodiscard]] auto& getLastOutput() { return _lastOutput; }
    [[nodiscard]] auto& getLastInput() { return _lastInput; }

    [[nodiscard]] u32 getNeuronCount() const { return _numberOfNeurons; }
    [[nodiscard]] u32 getPrevNeuronCount() const { return _lastNumberOfNeurons; }

    std::variant<Output, OclBuffer> activationDerivative(u32 batchSize) const {
        if constexpr (std::is_same_v<ComputePolicy, CpuEigenPolicy>) {
            return ComputePolicy::template activationDerivative<ActivationPolicy>(std::get<Output>(_lastOutput));
        } else {
            return ComputePolicy::template activationDerivative<ActivationPolicy>(std::get<OclBuffer>(_lastOutput), _numberOfNeurons, batchSize);
        }
    }
};


#endif
