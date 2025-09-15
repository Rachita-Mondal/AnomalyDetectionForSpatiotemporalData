## Robust Spatiotemporally Contiguous Anomaly Detection Using Tensor Decomposition

Anomaly detection in spatiotemporal data is a challenging problem encountered in a variety 
of applications, including video surveillance, medical imaging data, and urban traffic monitoring. Existing
anomaly detection methods focus mainly on point anomalies and cannot deal with temporal and spatial
dependencies that arise in spatio-temporal data. Tensor-based anomaly detection methods have been
proposed to address this problem. Although existing methods can capture dependencies across different
modes, they are primarily supervised and do not account for the specific structure of anomalies. Moreover,
these methods focus mainly on extracting anomalous features without providing any statistical confidence.
In this paper, we introduce an unsupervised tensor-based anomaly detection method that simultaneously
considers the sparse and spatiotemporally smooth nature of anomalies. The anomaly detection problem is
formulated as a regularized robust low-rank + sparse tensor decomposition where the total variation of the
tensor with respect to the underlying spatial and temporal graphs quantifies the spatiotemporal smoothness
of the anomalies. Once the anomalous features are extracted, we introduce a statistical anomaly scoring
framework that accounts for local spatio-temporal dependencies. The proposed framework is evaluated on
both synthetic and real data.

## Contributors
Rachita Mondal, Mert Indibi, Tapabrata Maiti, Selin Aviyente -- 2024-present
