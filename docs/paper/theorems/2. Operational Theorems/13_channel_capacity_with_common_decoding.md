# **Theorem 13** *(Channel Capacity with Common Decoding, under separation)*
Over a DMC with Shannon capacity $C_{\text{Shannon}}$, a common message decodable under every context has payload rate:
$$
R_{\text{payload}} = C_{\text{Shannon}} - K(P)
$$

with a strong converse. (see App. A.9, A.10)

**Proof Strategy:**

- *Achievability:* Split rate—payload at $C_{\text{Shannon}} - \varepsilon$ and witness at $K(P) + \varepsilon$, time-sharing or using shared randomness.
- *Converse:* A payload $> C_{\text{Shannon}} - K(P) + \varepsilon$ either exceeds channel capacity or underfunds the witness ($< K(P)$), contradicting Theorem 11.

With a **common-reconstruction** requirement, the encoder must pick **one** reproduction rule that all contexts accept—exactly the regime formalized by **Steinberg (2009)**.
