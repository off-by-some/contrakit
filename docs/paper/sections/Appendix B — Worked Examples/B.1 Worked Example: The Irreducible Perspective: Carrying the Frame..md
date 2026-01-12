## B.1 Worked Example: The Irreducible Perspective: Carrying the Frame.

Let's consider an example. Let's say there are three friends—*Nancy*, *Dylan*, *Tyler*—watch the coin from three seats: $\text{LEFT}$, $\text{MIDDLE}$, and $\text{RIGHT}$. After each flip they copy only the three words they saw, comma-separated, into a shared notebook. Each knew where they sat, and the order the games went, so they didn't think to write the positions.

Notebook

| **Flip** | **Observer Reports** |
| --- | --- |
| 1 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |
| 2 | $\text{NO}$,$\text{BOTH}$,$\text{YES}$ |
| 3 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |
| 4 | $\text{NO}$,$\text{BOTH}$,$\text{YES}$ |
| 5 | $\text{YES}$,$\text{BOTH}$,$\text{NO}$ |

Now, using only this notebook—can you tell who sat where? You can't. (Unless you're grading your own work, in which case: 10/10, excellent deduction).

It's because "$\text{YES}$, $\text{BOTH}$, $\text{NO}$" fits two different worlds on every line:

- On $\text{HEADS}$ the left seat is $\text{YES}$, the middle is $\text{BOTH}$, the right is $\text{NO}$;
- On $\text{TAILS}$ the left is $\text{NO}$, the middle is $\text{BOTH}$, the right is $\text{YES}$.

Strip names and seats, and those worlds collapse to the same record. You are exactly one yes/no short: "who wrote $\text{YES}$—left or right?" Formally, each report depends on two coordinates: the coin's face $S\in\{\text{HEADS},\text{TAILS}\}$ and the viewpoint $P\in\{L,M,R\}$.

The observation $O\in\{\text{YES},\text{NO},\text{BOTH}\}$ is

$$
O(S,P)= \begin{cases} \text{BOTH}, & P=M,\\ \text{YES}, & (S,P)\in\{(\text{HEADS},L),(\text{TAILS},R)\},\\ \text{NO}, & (S,P)\in\{(\text{HEADS},R),(\text{TAILS},L)\}. \end{cases}
$$

The smoking gun appears when we hide $P$: distinct worlds land on the same word.

- $(\text{HEADS},L)$ and $(\text{TAILS},R)$ both read $\text{YES}$;
- $(\text{HEADS},R)$ and $(\text{TAILS},L)$ both read $\text{NO}$.

There is no frame-independent reconstruction from $O$ alone. Every crisp report owes one more yes/no if you want the scene to be recoverable. Carry $P$, and the correlations are captured perfectly; drop it, and two different worlds become indistinguishable on the page.

Now quantify the coordination cost of omitting $P$. Suppose seats are used equally and the coin is fair, so $\text{YES}$, $\text{BOTH}$, $\text{NO}$ each appear about one-third of the time in the notebook.

- Seeing $\text{BOTH}$ pins $P=M$: no uncertainty remains (0 missing bits).
- Seeing $\text{YES}$ or $\text{NO}$ tells you only "not the middle," leaving a binary uncertainty—left or right (1 missing bit).

Averaging,

$$
\text{Missing}=\tfrac13\cdot 0+\tfrac13\cdot 1+\tfrac13\cdot 1=\tfrac23\ \text{bits per line}.
$$

In Shannon's notation, with $H(P)=\log_2 3$,

$$
I(O;P)=H(P)-H(P\mid O)=\log_2 3-\tfrac23\approx 0.918\ \text{bits},\qquad \boxed{H(P\mid O)=\tfrac23\ \text{bits per line}}.
$$

Plainly: unless you carry the frame, you are on average two-thirds of a bit short of knowing where each observer sat. Thus, the cost is small, but persistent.

It is not the one-off surprise of model identification; it is a steady coordination cost dictated by the object's geometry. If you also record the coin face $S$ alongside $O$, then together $(S,O)$ identify the seat exactly, driving this residual to zero—but at the price of carrying additional information with every observation. The lesson is not that perspective is noise to be averaged out, but that it is structure to be carried.

Keep the frame and the story is coherent. Drop it, and worlds become epistemically indistinct.