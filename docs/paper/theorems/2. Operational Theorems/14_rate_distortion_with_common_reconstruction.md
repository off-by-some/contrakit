# **Theorem 14** *(Rate-Distortion with Common Reconstruction, under separation)*
Under a common-reconstruction requirement across all contexts—
$$
R(D) = R_{\text{Shannon}}(D) + K(P)
$$

This is the **Steinberg** regime—our surcharge shifts the classical $R_{\text{Shannon}}(D)$ by **+K(P)** because the single reconstruction must harmonize incompatible frames (Steinberg, 2009) with a strong converse. Put differently—this establishes the distortion cost of consensus.

**Proof Strategy:**

- *Achievability:* Classical RD coding at $R_{\text{Shannon}}(D) + \varepsilon$ plus a $K(P) + \varepsilon$ witness enabling a single reconstruction rule for all contexts.
- *Converse:* d-tilted information converses imply any rate $< R_{\text{Shannon}}(D) + K(P) - \varepsilon$ forces a sub-$K(P)$ witness, contradicting Theorems 9–11.