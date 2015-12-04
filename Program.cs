using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new NeuralNetwork(4, 4, 1);   //input,hidden,output

            int trainingIterations = 10000;     //# of times to train neural network

            Console.WriteLine("Training Network...");
            for (int i = 0; i < trainingIterations; i++)
            {
                network.Train(0,0,0,0);     //send signal
                network.BackPropagate(1);   //calculate weights

                network.Train(1,1,1,1);
                network.BackPropagate(0);
            }

            double error;
            double output;

            output = network.Compute(0, 0, 0, 0)[0];
            error = network.CalculateError(1);
            Console.WriteLine("0 = 1 = " + output.ToString("F5") + ", Error = " + error.ToString("F5"));

            output = network.Compute(1, 1, 1, 1)[0];
            error = network.CalculateError(0);
            Console.WriteLine("1 = 0 = " + output.ToString("F5") + ", Error = " + error.ToString("F5"));

            output = network.Compute(0, 0, 0, 1)[0];
            error = network.CalculateError(0);
            Console.WriteLine("1 = 0 = " + output.ToString("F5") + ", Error = " + error.ToString("F5"));

        }
    }

    public class NeuralNetwork
    {
        public double LearnRate { get; set; }   //how fast they "learn", around 1 yields best results
        public double Momentum { get; set; }    //slows down "jumping"  variance in numbers
        public List<Neuron> InputLayer { get; set; }
        public List<Neuron> HiddenLayer { get; set; }
        public List<Neuron> OutputLayer { get; set; }
        static Random random = new Random();

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            LearnRate = .9;
            Momentum = .04;
            InputLayer = new List<Neuron>();
            HiddenLayer = new List<Neuron>();
            OutputLayer = new List<Neuron>();

            for (int i = 0; i < inputSize; i++)
                InputLayer.Add(new Neuron());

            for (int i = 0; i < hiddenSize; i++)
                HiddenLayer.Add(new Neuron(InputLayer));

            for (int i = 0; i < outputSize; i++)
                OutputLayer.Add(new Neuron(HiddenLayer));
        }

        public void Train(params double[] inputs)
        {
            int i = 0;
            InputLayer.ForEach(a => a.Value = inputs[i++]);
            HiddenLayer.ForEach(a => a.CalculateValue());
            OutputLayer.ForEach(a => a.CalculateValue());
        }

        public double[] Compute(params double[] inputs)
        {
            Train(inputs);
            return OutputLayer.Select(a => a.Value).ToArray();
        }

        public double CalculateError(params double[] targets)
        {
            int i = 0;
            return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
        }

        public void BackPropagate(params double[] targets)
        {
            int i = 0;
            OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            HiddenLayer.ForEach(a => a.CalculateGradient());
            HiddenLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
            OutputLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
        }

        public static double NextRandom()       
        {
            return 2 * random.NextDouble() - 1;
        }

        public static double SigmoidFunction(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double f)
        {
            return f * (1 - f);
        }
    }

    public class Neuron
    {
        public List<Synapse> InputSynapses { get; set; }
        public List<Synapse> OutputSynapses { get; set; }
        public double Bias { get; set; }
        public double BiasDelta { get; set; }
        public double Gradient { get; set; }
        public double Value { get; set; }

        public Neuron()
        {
            InputSynapses = new List<Synapse>();
            OutputSynapses = new List<Synapse>();
            Bias = NeuralNetwork.NextRandom();
        }

        public Neuron(List<Neuron> inputNeurons) : this()
        {
            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = new Synapse(inputNeuron, this);
                inputNeuron.OutputSynapses.Add(synapse);
                InputSynapses.Add(synapse);
            }
        }

        public virtual double CalculateValue()
        {
            return Value = NeuralNetwork.SigmoidFunction(InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value) + Bias);
        }

        public virtual double CalculateDerivative()
        {
            return NeuralNetwork.SigmoidDerivative(Value);
        }

        public double CalculateError(double target)
        {
            return target - Value;
        }

        public double CalculateGradient(double target)
        {
            return Gradient = CalculateError(target) * CalculateDerivative();
        }

        public double CalculateGradient()
        {
            return Gradient = OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * CalculateDerivative();
        }

        public void UpdateWeights(double learnRate, double momentum)
        {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient; // * 1
            Bias += BiasDelta + momentum * prevDelta;

            foreach (var s in InputSynapses)
            {
                prevDelta = s.WeightDelta;
                s.WeightDelta = learnRate * Gradient * s.InputNeuron.Value;
                s.Weight += s.WeightDelta + momentum * prevDelta;
            }
        }
    }

    public class Synapse
    {
        public Neuron InputNeuron { get; set; }
        public Neuron OutputNeuron { get; set; }
        public double Weight { get; set; }
        public double WeightDelta { get; set; }

        public Synapse(Neuron inputNeuron, Neuron outputNeuron)
        {
            InputNeuron = inputNeuron;
            OutputNeuron = outputNeuron;
            Weight = NeuralNetwork.NextRandom();
        }
    }
}