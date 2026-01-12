# 2. Building Intuition with the Lenticular Coin

Here, "contradiction" doesn't refer to the classical logical impossibility ($A$ and $¬A$), but rather to *epistemic incompatibility*: when two equally valid observations cannot be reconciled within a single reference frame ($A@X ∧ B@Y$ where $X$ and $Y$ represent incompatible contexts). This is similar to special relativity, where two observers can measure different times for the same event—and both are correct—because the reference frame fundamentally matters.

>  *"What we observe is not nature itself but nature exposed to our method of questioning."*
>
> — Werner Heisenberg, Physics and Philosophy: The Revolution in Modern Science

We can consider special relativity: two clocks read different times for the same event and both are right—because frame does the real work. This is exactly the measurement stance Heisenberg emphasized: what we observe is nature as interrogated by a method—in this case, the method is the viewing frame that accompanies the record. Lenticular images make this tactile. Tilt a postcard: from one angle you see one picture; from another, a different one. **The substrate doesn't change—your perspective does.**

If we apply that to a fair coin, then like Shannon's coin, it has two sides and we flip it at random. Unlike Shannon's coin however, each face is printed lenticularly so that what you see, depends on the viewing angle. We put the coin on the table with each face lenticularly printed, so the message you see depends on where you stand. We flip the coin. Since one person stands to the left, and the other to the right, when the coin lands, the left observer sees YES and the right observer sees NO; on the next flip those roles swap.

When they compare notes, they'll always disagree:

| Coin Side | LEFT Observer Sees | RIGHT Observer Sees |
| --------- | ------------------ | ------------------- |
| HEADS     | YES                | NO                  |
| TAILS     | NO                 | YES                 |

This is intuitive, that isn't a mistake or noise; it's baked into the viewing geometry. What happened depends on where you looked.

Formally we'd say: Let $S\in{\text{HEADS},\text{TAILS}}$ be the face up, $P$ the viewpoint (e.g., $\text{LEFT}$ or $\text{RIGHT}$), and let $O(S,P)\in{\text{YES},\text{NO}}$ be the visible message.

By design,
$$
O(S,P)= \begin{cases} \text{YES}, & (S,P)\in\{(\text{HEADS},\text{LEFT}),(\text{TAILS},\text{RIGHT})\},\\ \text{NO},  & (S,P)\in\{(\text{HEADS},\text{RIGHT}),(\text{TAILS},\text{LEFT})\}. \end{cases}
$$
We commence each trial as follows: flip the coin (fair, $1/2$–$1/2$), both observers record what they see, then compare notes. They always disagree. From either seat alone, the sequence looks like a fair binary source. Jointly, the outcomes are perfectly anti-correlated. While it remains true that what happens depended on where you were, this version still admits a single global description once we include $P$ in the state: the device implements a fixed rule ("$\text{LEFT}$ shows the opposite of $\text{RIGHT}$, with flip swapping roles"). Thus, this is anti-correlation, not an irreconcilable contradiction.