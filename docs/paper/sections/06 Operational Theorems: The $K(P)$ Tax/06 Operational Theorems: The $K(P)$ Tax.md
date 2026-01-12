## 6. Operational Theorems: The $K(P)$ Tax

All rates are in bits. Intermediate Rényi/Chernoff expressions use natural logs; convert via $\log_2 x = (\ln x)/\ln 2$.

$$
\alpha^\star(P)=\min_{\lambda\in\Delta(\mathcal C)}\max_{Q\in\mathrm{FI}}\sum_c \lambda_c\,\mathrm{BC}(p_c,q_c),
\qquad
K(P)=-\log_2 \alpha^\star(P).
$$

The structural number $K(P) = -\log_2 \alpha^\star(P)$ from previous sections now becomes operational—it manifests as an exact tax in every information-theoretic task involving multiple contexts. Each theorem shows the same pattern: whatever the baseline rate would be in classical Shannon theory, **contradiction adds exactly $K(P)$ bits per symbol**.

The mechanism is simple: a **witness** is a short string of rate $K(P)$ that certifies how contexts must coordinate to maintain consistency. When witnesses are adequately funded, all decoders can agree on the reconstruction; when underfunded, some decoder must fail. As always, please assume finite alphabets and FI is convex/compact; product-closure for product claims; source–channel separation where stated.

---

## Reader's Guide to Main Results

**Core Information Theory (§6.1-6.5):**

- **Theorem 6:** Typical sets for $(X^n, W_n)$ have size $2^{n(H(X|C) + K(P))}$ (see Appendix A.3.2, Appendix A.2.2, Appendix A.5.1, Appendix A.9)
- **Theorems 7-8:** Compression rates are $H(X|C) + K(P)$ (known contexts) or $H(X) + K(P)$ (latent) (see Appendix A.9)
- **Theorem 9:** Testing against frame-independence requires type-II exponent $\ge K(P)$ (see Appendix A.3.2, Appendix A.9)
- **Theorem 10:** Eliminating contradiction needs witness rate $\ge K(P)$ (achievability via Appendix A.12; TV lower bound cf. Appendix A.11)

**Multi-Context Communication (§6.6-6.8):**

- **Theorem 11:** Common messages decodable by all contexts cost $H(X|C) + K(P)$ (see Appendix A.9)
- **Theorem 12:** Any common representation carries $\ge H(X|C) + K(P)$ bits per symbol (see Appendix A.9, Appendix A.10)
- **Theorems 13-14:** Channel capacity and rate-distortion both lose exactly $K(P)$ (see Appendix A.9, Appendix A.10)

**Geometric Structure (§6.9):**

- **Theorem 15:** Hellinger geometry explains why contradiction costs compose linearly in $K$ (and subadditively in angle) (Appendix A.2.2, Appendix A.10; FI product closure Appendix A.1.8)