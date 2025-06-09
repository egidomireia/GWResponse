# GWResponse
## LISA Response Function

This repository implements a simplified, frequency-domain model of the gravitational wave (GW) response of the LISA mission. The goal is to simulate the detector response to signals from compact binary coalescences, using the (2,2) mode of the **IMRPhenomD** waveform and leveraging **JAX** for efficient computation and differentiation.

The model is based on the formalism of [arXiv:2003.00357](https://arxiv.org/abs/2003.00357).

### Assumptions

To focus on the core GW response, the following simplifications are adopted:

- No relativistic corrections of order v/c, including Doppler effects.
- Flat spacetime: no gravitational redshift or light deflection from the Sun.
- Rigid, equilateral triangle geometry for LISA, with constant armlengths.
- Simultaneous evaluation of all geometric quantities (no inter-spacecraft light travel delays).

These approximations are sufficient for simulating the response in the low-noise band of LISA, and can be refined later to include more realistic effects.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/egidomireia/GWResponse.git
pip install -r requirements.txt
```

Or use the setup script:
```bash
pip install .
```

To import the response module, use:
```bash
from gwresponse.response import LISA_response
```
