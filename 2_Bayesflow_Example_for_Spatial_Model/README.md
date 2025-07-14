# LGCP Inference with BayesFlow and Zuko

This foler implements simulation-based inference for a **Latent Gaussian Cox Process (LGCP)** model using a **Convolutional Summary Network** and **Neural Spline Flows (NSF)** from the [Zuko](https://github.com/probabilists/zuko) library. The inference approach follows the BayesFlow paradigm, enabling amortized posterior estimation of model parameters from spatial count data.

---

## 🧠 Key Features

- **LGCP Simulator**: Simulates spatial Poisson count data using exponential covariance Gaussian Processes.
- **Masking Mechanism**: Emulates partial observation by masking portions of the 2D spatial domain.
- **CNN Summary Network**: Learns to encode masked 2D data into compact feature vectors for inference.
- **Neural Spline Flow**: Estimates posterior distributions over parameters via normalizing flows (Zuko).
- **Training Pipeline**: End-to-end amortized training using PyTorch.
- **Visualization**: Posterior plots, parameter recovery curves, and predictive samples.

---

## 📦 Dependencies

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- [Zuko](https://github.com/probabilists/zuko)

Install dependencies via:

```bash
pip install torch numpy scipy matplotlib zuko
```

---

## 📚 References

Here are some resources on the key concepts used in this project:

### BayesFlow & Simulation-Based Inference (SBI)
BayesFlow is a framework for simulation-based inference that uses normalizing flows to learn the posterior distribution of simulator parameters. It enables amortized inference, meaning that once the model is trained, it can rapidly compute the posterior for new observations without re-training.

- **Paper:** Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). *BayesFlow: Learning Complex Stochastic Models with Invertible Neural Networks*. [arXiv:2003.06281](https://arxiv.org/abs/2003.06281)


### Neural Spline Flows
Neural Spline Flows are a type of normalizing flow that uses monotonic rational-quadratic splines to create flexible and expressive invertible transformations. This allows for the modeling of complex, multimodal distributions. They are a key component of libraries like `Zuko`.

- **Paper:** Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). *Neural Spline Flows*. [arXiv:1906.04032](https://arxiv.org/abs/1906.04032)
- **Library:** The `Zuko` library implements Neural Spline Flows. [GitHub Repository](https://github.com/probabilists/zuko)
