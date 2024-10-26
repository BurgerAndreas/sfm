# Distributions

## Gaussian Distribution
The log probability (log likelihood) of a multivariate Gaussian (normal) distribution is given by the following formula:
mean=0, variance=1, covariance=0

\[
\log P(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{1}{2} \left( (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) + \log |\boldsymbol{\Sigma}| + k \log 2\pi \right)
\]

Where:
- \(\mathbf{x}\) is the data point (a vector).
- \(\boldsymbol{\mu}\) is the mean vector of the Gaussian distribution.
- \(\boldsymbol{\Sigma}\) is the covariance matrix.
- \(k\) is the dimensionality of \(\mathbf{x}\).
- \(|\boldsymbol{\Sigma}|\) denotes the determinant of the covariance matrix \(\boldsymbol{\Sigma}\).

1. The first term \((\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\) is the Mahalanobis distance, which measures the distance of \(\mathbf{x}\) from the mean \(\boldsymbol{\mu}\) in the space defined by \(\boldsymbol{\Sigma}\).
2. The second term \(\log |\boldsymbol{\Sigma}|\) accounts for the normalization factor related to the covariance matrix.
3. The third term \(k \log 2\pi\) normalizes the probability over the \(k\)-dimensional space.

If the mean and the variance are the same along every dimension, the multivariate Gaussian distribution becomes an isotropic Gaussian. In this case:

- The mean vector \(\boldsymbol{\mu}\) remains the same for each dimension, so \(\boldsymbol{\mu} = (\mu, \mu, \dots, \mu)\).
- The covariance matrix \(\boldsymbol{\Sigma}\) becomes \(\sigma^2 I\), where \(I\) is the identity matrix, meaning that the variance is the same (\(\sigma^2\)) for all dimensions and there is no correlation between different dimensions.

Under these conditions, the log probability simplifies considerably:

### Isotropic Gaussian
mean and variance same for all dimensions, covariance=0

\[
\log P(\mathbf{x}|\boldsymbol{\mu}, \sigma^2) = -\frac{1}{2} \left( \frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{\sigma^2} + k \log(2 \pi \sigma^2) \right)
\]

Where:
- \(\mathbf{x}\) is the data point (a vector).
- \(\boldsymbol{\mu} = (\mu, \mu, \dots, \mu)\) is the mean vector, where all elements are equal to \(\mu\).
- \(\sigma^2\) is the variance, the same for all dimensions.
- \(k\) is the number of dimensions.
- \(\|\mathbf{x} - \boldsymbol{\mu}\|^2\) is the squared Euclidean distance between \(\mathbf{x}\) and the mean vector.

1. Mahalanobis distance simplifies: 
   - In the general case, you have the term \((\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\), but with \(\boldsymbol{\Sigma} = \sigma^2 I\), the inverse of \(\boldsymbol{\Sigma}\) becomes \(\frac{1}{\sigma^2} I\), leading to \(\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{\sigma^2}\).
   
2. Log determinant simplifies:
   - The determinant of \(\boldsymbol{\Sigma}\), which is \(|\boldsymbol{\Sigma}| = \sigma^{2k}\), simplifies the log term \(\log |\boldsymbol{\Sigma}|\) to \(k \log \sigma^2\).
   
3. Normalization term: 
   - The general normalization term \(\log |\boldsymbol{\Sigma}| + k \log 2\pi\) simplifies to \(k \log(2\pi\sigma^2)\).

Thus, with isotropic Gaussian, the log probability formula becomes simpler due to uniform variance and no covariance between dimensions.

When the mean \(\mu = 0\) and the standard deviation \(\sigma = 1\), the Gaussian distribution becomes the standard normal distribution. For this case, both the mean and variance are fixed, and the covariance matrix \(\boldsymbol{\Sigma} = I\) (the identity matrix).

For a univariate Gaussian distribution (single dimension) where \(\mu = 0\) and \(\sigma = 1\), the probability density function (PDF) simplifies to:

\[
P(x | \mu = 0, \sigma = 1) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
\]

Taking the log of this PDF gives us the log probability for the standard normal distribution:

### Log Probability for a Univariate Standard Normal Distribution:
\[
\log P(x | \mu = 0, \sigma = 1) = -\frac{x^2}{2} - \frac{1}{2} \log(2\pi)
\]

For a multivariate standard normal distribution in \(k\) dimensions, where the mean vector \(\boldsymbol{\mu} = 0\) (all zeros) and the covariance matrix \(\boldsymbol{\Sigma} = I\) (identity matrix), the log probability becomes:

### Log Probability for a Multivariate Standard Normal Distribution:
\[
\log P(\mathbf{x} | \boldsymbol{\mu} = 0, \boldsymbol{\Sigma} = I) = -\frac{1}{2} \left( \|\mathbf{x}\|^2 + k \log(2\pi) \right)
\]

Where:
- \(\mathbf{x}\) is the data point (vector).
- \(\|\mathbf{x}\|^2\) is the squared Euclidean norm of \(\mathbf{x}\) (i.e., the sum of squares of the elements of \(\mathbf{x}\)).
- \(k\) is the number of dimensions.

1. \(-\frac{1}{2} \|\mathbf{x}\|^2\): This is the term that arises from the Mahalanobis distance, which is just the squared Euclidean distance from the origin since the mean is zero and the covariance matrix is the identity.
2. \(-\frac{k}{2} \log(2\pi)\): This is the normalization term, simplified because the variance is 1 in every dimension.

Thus, the log probability for the standard normal distribution is much simpler since both the mean and variance are fixed.

## Beta Distribution
Symmetric beta distributions with larger parameter values are closer to Gaussian.
beta distributions are on (0,1) while all Gaussian distributions are on (−∞,∞)
The standardized beta(4,4) is restricted to lie between -3 and 3 (the standard Gaussian can take any value); it is also less peaked than the Gaussian


