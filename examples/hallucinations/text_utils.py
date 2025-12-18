"""
Text display utilities for mathematical hypothesis testing of hallucination inevitability.

This module provides comprehensive display and formatting utilities for experiments
testing whether logical contradictions in training data force language models and
neural networks to hallucinate when answering impossible questions.

The experiments probe mathematical hypotheses from contradiction theory:
- Tasks with contradiction measure K > 0 force hallucination rates ≥ 1-2^(-K)
- Observed hallucination rates depend on K plus architectural constraints
- Statistical significance is tested against theoretical lower bounds

Key components:
- Rich console display system with color-coded results
- Experiment result cards showing theory vs observation
- Statistical analysis including confidence intervals and p-values
- Confidence calibration and architectural comparison tools

Assumptions:
- Results are displayed using Rich for terminal formatting
- Statistical tests use binomial confidence intervals and exact p-values
- Confidence intervals assume independent trials with fixed probability

Typical usage:
- Import display functions in experiment scripts: from text_utils import display
- Use display.print_experiment_card() to show comprehensive results
- Statistical tests automatically compute significance against theoretical bounds
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.rule import Rule
from rich.align import Align

if TYPE_CHECKING:
    from experiment_ollama import ExperimentResults


class Verbosity:
    """How much detail to show in the output."""
    QUIET = 0     # Only final results and key findings
    NORMAL = 1    # Progress updates and main measurements
    DEBUG = 2     # Everything including raw model responses and all errors


VERBOSITY_LEVEL = Verbosity.NORMAL

# ==============================================================================
# DISPLAY SYSTEM
# ==============================================================================

class Display:
    """Centralized display system using Rich for beautiful console output."""
    def __init__(self, verbosity_level: Optional[int] = None):
        self.console = Console()
        # Use global verbosity level if not specified
        self.verbosity_level = verbosity_level if verbosity_level is not None else VERBOSITY_LEVEL
        self._active_progress_bars = []
    
    @staticmethod
    def _color_for_threshold(value: float, thresholds: Tuple[float, float] = (0.1, 0.3)) -> str:
        """Determine color based on threshold values (green/yellow/red)."""
        if value > thresholds[1]:
            return "red"
        elif value > thresholds[0]:
            return "yellow"
        return "green"
    
    @staticmethod
    def _color_for_delta(delta: float) -> str:
        """Determine color for delta values (positive=red, negative=green, zero=yellow)."""
        if delta > 0:
            return "red"
        elif delta < 0:
            return "green"
        return "yellow"
    
    @staticmethod
    def _status_icon_and_color(condition: bool) -> Tuple[str, str]:
        """Get status icon and color based on condition."""
        if condition:
            return ("✓", "green")
        return ("✗", "red")
    
    def _append_labeled_value(self, text: Text, label: str, value: str, value_style: str = "", note: str = "", note_style: str = "dim") -> None:
        """Append a labeled value to Text object with optional note."""
        text.append(f"  {label}: ", style="bold")
        if value_style:
            text.append(value, style=value_style)
        else:
            text.append(value)
        if note:
            text.append(f" {note}\n", style=note_style)
        else:
            text.append("\n")
    
    def _append_percentage(self, text: Text, label: str, value: float, precision: int = 0, value_style: str = "", note: str = "", note_style: str = "dim") -> None:
        """Append a percentage value to Text object."""
        format_str = f"{{value:.{precision}%}}"
        formatted_value = format_str.format(value=value)
        self._append_labeled_value(text, label, formatted_value, value_style, note, note_style)
    
    def _append_confidence_interval(self, text: Text, ci_lower: float, ci_upper: float, n_trials: int, style: str = "dim") -> None:
        """Append confidence interval information to Text object."""
        ci_width = ci_upper - ci_lower
        if ci_width > 0:
            text.append(f" ± {ci_width/2:.1%}", style=style)
            text.append(f" (N={n_trials}, 95% CI)\n", style=style)
        else:
            text.append(f" (N={n_trials})\n", style=style)
    
    @staticmethod
    def _format_confidence_stats(confidences: List[float], precision: int = 0) -> str:
        """Format confidence statistics (mean, range, std) as a string."""
        if not confidences:
            return "N/A"
        mean_val = np.mean(confidences)
        min_val = min(confidences)
        max_val = max(confidences)
        std_val = np.std(confidences)
        format_str = f"{{val:.{precision}%}}"
        return f"mean=[cyan]{format_str.format(val=mean_val)}[/cyan], range=[cyan]{format_str.format(val=min_val)}-{format_str.format(val=max_val)}[/cyan], std=[cyan]{format_str.format(val=std_val)}[/cyan]"
    
    @staticmethod
    def _format_percentage(value: float, precision: int = 0) -> str:
        """Format a percentage value with specified precision."""
        return f"{value:.{precision}%}"
    
    @staticmethod
    def _format_percentage_range(min_val: float, max_val: float, precision: int = 0) -> str:
        """Format a percentage range."""
        format_str = f"{{val:.{precision}%}}"
        return f"{format_str.format(val=min_val)}-{format_str.format(val=max_val)}"
    
    def _append_formatted_percentage(self, text: Text, label: str, value: float, precision: int = 0, 
                                     value_style: str = "", note: str = "", note_style: str = "dim") -> None:
        """Append a formatted percentage with label to Text object."""
        formatted = self._format_percentage(value, precision)
        self._append_labeled_value(text, label, formatted, value_style, note, note_style)
    
    def _append_confidence_stats(self, text: Text, label: str, confidences: List[float], 
                                 precision: int = 0, note: str = "", note_style: str = "dim") -> None:
        """Append confidence statistics to Text object."""
        if not confidences:
            self._append_labeled_value(text, label, "N/A", note=note, note_style=note_style)
            return
        stats_str = self._format_confidence_stats(confidences, precision)
        text.append(f"  {label}: ", style="bold")
        text.append(stats_str)
        if note:
            text.append(f" {note}", style=note_style)
        text.append("\n")

    def should_show(self, level: int) -> bool:
        """Check if content at given verbosity level should be displayed."""
        return self.verbosity_level >= level
    
    def register_progress_bar(self, pbar) -> None:
        """Register a progress bar to be managed during verbose output."""
        if pbar is not None:
            self._active_progress_bars.append(pbar)
    
    def unregister_progress_bar(self, pbar) -> None:
        """Unregister a progress bar."""
        if pbar is not None and pbar in self._active_progress_bars:
            self._active_progress_bars.remove(pbar)
    
    def _hide_progress_bars(self) -> None:
        """Temporarily hide all registered progress bars."""
        for pbar in self._active_progress_bars:
            if hasattr(pbar, 'clear'):
                pbar.clear(nolock=True)
    
    def _show_progress_bars(self) -> None:
        """Restore all registered progress bars."""
        for pbar in self._active_progress_bars:
            if hasattr(pbar, 'refresh'):
                pbar.refresh()
    
    def _print_with_progress_handling(self, content: str, method: str = "print") -> None:
        """Print content while managing progress bars."""
        self._hide_progress_bars()
        getattr(self.console, method)(content)
        self._show_progress_bars()
    
    def _without_progress_bars(self):
        """Context manager to temporarily hide progress bars."""
        class ProgressBarContext:
            def __init__(self, display):
                self.display = display
            def __enter__(self):
                self.display._hide_progress_bars()
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.display._show_progress_bars()
        return ProgressBarContext(self)

    def print_section_header(self, title: str) -> None:
        """Print a major section header with a decorative rule."""
        with self._without_progress_bars():
            self.console.print()
            self.console.print(Rule(f"[bold cyan]{title}", style="cyan"))

    def print_subsection_header(self, title: str) -> None:
        """Print a subsection header."""
        self._print_with_progress_handling(f"\n[bold magenta]{title}[/bold magenta]")

    def print_table(self, headers: List[str], rows: List[List[str]], title: Optional[str] = None) -> None:
        """Print a table with optional title."""
        if not rows:
            return

        with self._without_progress_bars():
            table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD)
            for header in headers:
                table.add_column(header)

            for row in rows:
                table.add_row(*[str(cell) for cell in row])

            if title:
                self.console.print(Panel(table, title=title, border_style="blue"))
            else:
                self.console.print(table)

    def print_experiment_card(self, result: "ExperimentResults", verbosity: int = Verbosity.NORMAL) -> None:
        """Print a comprehensive experiment results card."""
        if not self.should_show(verbosity):
            return

        with self._without_progress_bars():
            status_icon, status_color = self._status_icon_and_color(
                result.observed_hallucination_rate >= result.lower_bound
            )

            card_content = Text()

            # Theory Prediction Section
            card_content.append("WHAT THEORY PREDICTS\n", style="bold yellow")
            self._append_percentage(
                card_content, "Minimum hallucination", result.lower_bound, 
                precision=1, value_style="yellow",
                note="(must be at least this due to task contradiction)"
            )

            card_content.append("  Output choice entropy: ", style="bold")
            card_content.append(f"+{result.decision_entropy:.2f} bits", style="yellow")
            card_content.append(" (from having 7 possible weekdays to choose from)\n", style="dim")

            card_content.append("  Uncertainty expression: ", style="bold")
            card_content.append(f"r≈{result.witness_capacity:.2f} bits", style="yellow")
            capacity_note = " (model cannot express uncertainty)" if result.witness_capacity < 0.01 else " (model can express some uncertainty)"
            card_content.append(capacity_note + "\n", style="dim")

            # Observed Section
            card_content.append("\nOBSERVED\n", style="bold cyan")
            card_content.append("  Hallucination: ", style="bold")
            card_content.append(f"{result.observed_hallucination_rate:.1%}", style="magenta")
            self._append_confidence_interval(
                card_content, 
                result.hallucination_rate_ci_lower, 
                result.hallucination_rate_ci_upper, 
                result.n_trials
            )

            card_content.append("  Confidence: ", style="bold")
            card_content.append(f"{result.average_confidence:.0%}", style="cyan")
            if result.confidence_std > 0:
                card_content.append(f" (σ={result.confidence_std:.0%}, range: {result.confidence_min:.0%}-{result.confidence_max:.0%})", style="dim")
            card_content.append("\n")

            excess = result.observed_hallucination_rate - result.lower_bound
            excess_color = self._color_for_threshold(excess, (0.1, 0.3))
            card_content.append("  Extra hallucination: ", style="bold")
            card_content.append(f"+{excess:.1%}", style=excess_color)
            card_content.append(" (beyond theoretical minimum, due to model architecture)\n", style="dim")

            # Validation Section
            card_content.append("\nVALIDATION\n", style="bold green")
            card_content.append(f"  {status_icon} ", style=status_color)
            card_content.append("Exceeds bound", style=status_color)

            # Show p-value if significant
            if result.p_value_exceeds_bound < 0.05:
                card_content.append(f" (p={result.p_value_exceeds_bound:.3f})", style="green")
            elif result.p_value_exceeds_bound < 0.10:
                card_content.append(f" (p={result.p_value_exceeds_bound:.3f})", style="yellow")
            card_content.append("\n")

            card_content.append("  ✓ Contradiction forces the model to hallucinate\n", style="green")
            card_content.append("  ✓ Model is confident despite impossible task\n", style="green")

            # Interpretation
            card_content.append("\nWHAT THIS MEANS\n", style="bold magenta")
            card_content.append(f"  {result.K:.2f} bits of contradiction make wrong answers mathematically unavoidable\n", style="dim")
            card_content.append("  Extra hallucination comes from having 7 possible answers to choose from\n", style="dim")
            card_content.append("  Model appears confident despite the impossible task\n", style="dim")

            panel = Panel(
                card_content,
                title=f"RESULT: {result.K:.2f} bits of task contradiction",
                border_style=status_color,
                padding=(0, 2)
            )
            self.console.print(panel)

    def print_comparison_panel(self, comparison_text: Text, title: str, border_style: str = "blue") -> None:
        """Print a comparison panel with formatted text."""
        with self._without_progress_bars():
            panel = Panel(
                comparison_text,
                title=title,
                border_style=border_style,
                padding=(0, 2)
            )
            self.console.print()
            self.console.print(panel)

    def print_simple_text(self, text: str) -> None:
        """Print simple text."""
        self._print_with_progress_handling(text)

    def print_simple_text_dim(self, text: str) -> None:
        """Print dimmed text."""
        self._print_with_progress_handling(f"[dim]{text}[/dim]")

    def print_colored_text(self, text: str, color: str) -> None:
        """Print text with specified color."""
        self._print_with_progress_handling(f"[{color}]{text}[/{color}]")

    def print_newline(self) -> None:
        """Print a newline."""
        with self._without_progress_bars():
            self.console.print()

    def print_task_properties(self, K: float, alpha_star: float, lower_bound: float, is_frame_independent: bool) -> None:
        """Print formatted task properties with key metrics."""
        self.print_simple_text(f"  [bold]Task contradiction[/bold] = [cyan]{K:.4f}[/cyan] bits  │  "
                     f"[bold]Best possible strategy[/bold] = [cyan]{alpha_star:.4f}[/cyan]  │  "
                     f"[bold]Minimum hallucination[/bold] = [yellow]{lower_bound:.2%}[/yellow]  │  "
                     f"Mathematically consistent: [green]{is_frame_independent}[/green]")

    def format_delta(self, delta: float, precision: int = 1) -> str:
        """Format a difference value with appropriate color coding."""
        color = self._color_for_delta(delta)
        sign = "+" if delta > 0 else ""
        return f"[{color}]{sign}{self._format_percentage(delta, precision)}[/{color}]"
    
    def _append_delta(self, text: Text, label: str, delta: float, precision: int = 1, 
                      note: str = "", note_style: str = "dim") -> None:
        """Append a delta value with color coding to Text object."""
        text.append(f"  {label}: ", style="bold")
        text.append(self.format_delta(delta, precision))
        if note:
            text.append(f" {note}", style=note_style)
        text.append("\n")

    def print_experiment_result_panel(self, result: "ExperimentResults", task_index: Optional[int] = None) -> None:
        """Print a formatted panel showing experiment results for a single task."""
        # Special handling for K=0 control
        is_control = result.K < 0.01

        if is_control:
            # K=0 control: should have 0% hallucination
            status_icon, status_color = self._status_icon_and_color(
                result.observed_hallucination_rate < 0.1
            )
            status_text = "BASELINE" if result.observed_hallucination_rate < 0.1 else "UNEXPECTED"
        else:
            # K>0: should exceed bound
            status_icon, status_color = self._status_icon_and_color(
                result.observed_hallucination_rate >= result.lower_bound
            )
            status_text = "CONFIRMED" if result.observed_hallucination_rate >= result.lower_bound else "UNEXPECTED"

        result_text = Text()
        if task_index is not None:
            result_text.append(f"Task {task_index}: ", style="bold white")

        if is_control:
            result_text.append(f"{result.n_contexts} context (K=0 CONTROL)\n", style="cyan bold")
            result_text.append("  Task contradiction: ", style="bold")
            result_text.append(f"{result.K:.2f} bits", style="green")
            result_text.append(" (no contradiction = hallucination should be ~0%)\n", style="dim")
            result_text.append("  Theory predicts: ", style="bold")
            result_text.append("~0%", style="green")
            result_text.append(" hallucination (task has unique correct answer)\n", style="dim")
        else:
            result_text.append(f"{result.n_contexts} conflicting contexts\n", style="cyan")
            result_text.append("  Task contradiction: ", style="bold")
            result_text.append(f"{result.K:.2f} bits", style="yellow")
            result_text.append(" (higher = more impossible)\n", style="dim")
            result_text.append("  Theory predicts: ", style="bold")
            result_text.append("at least ", style="dim")
            result_text.append(self._format_percentage(result.lower_bound), style="yellow")
            result_text.append(" hallucination rate\n", style="dim")

        result_text.append("  We observed: ", style="bold")
        result_text.append(self._format_percentage(result.observed_hallucination_rate), style="magenta")
        result_text.append(f" (N={result.n_trials})", style="dim")

        # Show CI
        ci_width = result.hallucination_rate_ci_upper - result.hallucination_rate_ci_lower
        if ci_width > 0:
            result_text.append(f" ± {self._format_percentage(ci_width/2, precision=0)}", style="dim")

        result_text.append(" hallucination  ", style="dim")
        result_text.append(f"{status_icon} {status_text}", style=status_color)

        # Show fabrications vs abstentions
        result_text.append(f"\n  Fabrications: {result.confident_fabrications}/{result.n_trials}, ", style="dim")
        result_text.append(f"Abstentions: {result.partial_abstentions}/{result.n_trials}", style="dim")

        with self._without_progress_bars():
            panel = Panel(
                result_text,
                border_style=status_color,
                padding=(0, 1)
            )
            self.console.print(panel)
        self.print_newline()

    def print_experiment_summary(self, results_list: List["ExperimentResults"]) -> None:
        """Print a comprehensive summary of all experiment results."""
        # Separate K=0 control from K>0 tasks
        control_results = [r for r in results_list if r.K < 0.01]
        contradiction_results = [r for r in results_list if r.K >= 0.01]

        summary_text = Text()

        # K=0 Control Results
        if control_results:
            control = control_results[0]
            summary_text.append("K=0 Control (No Contradiction):\n", style="bold green")
            summary_text.append(f"  Hallucination: {self._format_percentage(control.observed_hallucination_rate)}", style="green")
            if control.observed_hallucination_rate < 0.1:
                summary_text.append(" ✓ Near zero as expected\n", style="green")
            else:
                summary_text.append(" ⚠ Unexpectedly high (should be ~0%)\n", style="yellow")
            summary_text.append(f"  Fabrications: {control.confident_fabrications}/{control.n_trials}, ", style="dim")
            summary_text.append(f"Abstentions: {control.partial_abstentions}/{control.n_trials}\n\n", style="dim")

        # K>0 Results
        if contradiction_results:
            k_min = min(r.K for r in contradiction_results)
            k_max = max(r.K for r in contradiction_results)
            obs_min = min(r.observed_hallucination_rate for r in contradiction_results)
            obs_max = max(r.observed_hallucination_rate for r in contradiction_results)

            summary_text.append("K>0 Contradiction Tasks:\n", style="bold yellow")
            summary_text.append(f"  K Range: {k_min:.2f} → {k_max:.2f} bits\n", style="cyan")
            summary_text.append(f"  Hallucination: {self._format_percentage(obs_min)} → {self._format_percentage(obs_max)}\n", style="magenta")

            # Check if all exceeded bound
            all_exceeded = all(r.observed_hallucination_rate >= r.lower_bound for r in contradiction_results)
            if all_exceeded:
                summary_text.append(f"  ✓ All {len(contradiction_results)} tasks exceeded theoretical bound\n", style="green")
            else:
                n_exceeded = sum(1 for r in contradiction_results if r.observed_hallucination_rate >= r.lower_bound)
                summary_text.append(f"  ~ {n_exceeded}/{len(contradiction_results)} tasks exceeded bound\n", style="yellow")

            # Check for ceiling effect
            obs_range = obs_max - obs_min
            if obs_min > 0.95:
                summary_text.append("\n  ⚠ CEILING EFFECT DETECTED\n", style="yellow bold")
                summary_text.append("  All tasks saturated at ~100% hallucination\n", style="yellow")
                summary_text.append("  Cannot measure K relationship due to saturation\n", style="dim")
                summary_text.append("  Increase N_TRIALS for better resolution\n", style="dim")
            elif obs_range < 0.15:
                summary_text.append("\n  ⚠ LIMITED VARIATION DETECTED\n", style="yellow bold")
                summary_text.append(f"  Only {self._format_percentage(obs_range)} range across tasks\n", style="yellow")
                summary_text.append("  Consider: More trials or wider K range\n", style="dim")
            else:
                summary_text.append("\n  ✓ Clear K → hallucination relationship\n", style="green bold")
                summary_text.append(f"  {self._format_percentage(obs_range)} increase from K={k_min:.2f} to K={k_max:.2f}\n", style="green")

        with self._without_progress_bars():
            panel = Panel(summary_text, title="SUMMARY", border_style="green", padding=(0, 1))
            self.console.print(panel)
            self.console.print()

    def print_panel(self, content: Union[str, Text], title: Optional[str] = None, border_style: str = "blue", **kwargs) -> None:
        """Print a panel with the given content."""
        with self._without_progress_bars():
            panel = Panel(content, title=title, border_style=border_style, **kwargs)
            self.console.print(panel)


# Global display instance
display = Display()

# ==============================================================================
# OUTPUT FORMATTING
# ==============================================================================

def should_show(level: int) -> bool:
    """Check if content at given verbosity level should be displayed."""
    return VERBOSITY_LEVEL >= level


def print_section_header(title: str, verbosity: int = VERBOSITY_LEVEL) -> None:
    """Print a major section header using Rich Rule for clear visual separation."""
    if not should_show(verbosity):
        return
    display.print_section_header(title)


def print_subsection_header(title: str, verbosity: int = VERBOSITY_LEVEL) -> None:
    """Print a subsection header."""
    if not should_show(verbosity):
        return
    display.print_subsection_header(title)


def print_table(headers: List[str], rows: List[List[str]], title: Optional[str] = None,
                verbosity: int = VERBOSITY_LEVEL) -> None:
    """Print a table using Rich."""
    if not should_show(verbosity):
        return
    display.print_table(headers, rows, title)


def print_experiment_card(result: "ExperimentResults", verbosity: int = Verbosity.NORMAL) -> None:
    """Print a comprehensive experiment results card with full story."""
    display.print_experiment_card(result, verbosity)


def print_architectural_comparison(results_abstain: "ExperimentResults", results_forced: "ExperimentResults") -> None:
    """Print detailed architectural comparison between abstention and forced choice."""

    rate_delta = results_forced.observed_hallucination_rate - results_abstain.observed_hallucination_rate
    witness_delta = results_forced.witness_capacity - results_abstain.witness_capacity

    comparison_text = Text()
    comparison_text.append("Response Analysis:\n", style="bold")

    # Abstention condition
    comparison_text.append("  Abstention Allowed:\n", style="cyan bold")
    if results_abstain.partial_abstentions > 0:
        abstention_rate = results_abstain.partial_abstentions / results_abstain.n_trials
        comparison_text.append(f"    ✓ Abstentions: {display._format_percentage(abstention_rate)} ({results_abstain.partial_abstentions}/{results_abstain.n_trials})\n", style="green")
        comparison_text.append(f"    ✗ Fabrications: {display._format_percentage(results_abstain.observed_hallucination_rate)}\n", style="yellow")
    else:
        comparison_text.append(f"    ✗ Fabrications: {display._format_percentage(results_abstain.observed_hallucination_rate)}\n", style="red")
    comparison_text.append(f"    Uncertainty expression: {results_abstain.witness_capacity:.2f} bits\n", style="dim")

    # Forced condition
    comparison_text.append("\n  Forced Choice:\n", style="magenta bold")
    comparison_text.append(f"    ✗ Fabrications: {display._format_percentage(results_forced.observed_hallucination_rate)} (no abstention allowed)\n", style="red")
    comparison_text.append(f"    Uncertainty expression: {results_forced.witness_capacity:.2f} bits\n", style="dim")

    # Delta analysis
    comparison_text.append("\nArchitectural Effect:\n", style="bold")
    comparison_text.append("  Hallucination change: ", style="bold")
    comparison_text.append(display.format_delta(rate_delta, precision=1))
    comparison_text.append("\n")
    comparison_text.append(f"  Uncertainty expression change: {witness_delta:+.2f} bits\n", style="cyan")

    interpretation = ""
    if abs(rate_delta) < 0.05:
        interpretation = "Forcing doesn't change behavior (model ignores abstention option)"
    elif rate_delta > 0:
        interpretation = "Forcing increases hallucination (model would abstain if allowed)"
    else:
        interpretation = "Abstention helps reduce hallucination"

    comparison_text.append(f"\n  {interpretation}", style="dim italic")

    display.print_comparison_panel(comparison_text, "ABSTENTION VS FORCED CHOICE", "blue")


def format_architectural_comparison_text(
    halluc_abstain: float,
    halluc_forced: float
) -> Text:
    """
    Format text comparing abstention vs forced choice results.
    
    Args:
        halluc_abstain: Hallucination rate when abstention is allowed
        halluc_forced: Hallucination rate when forced to choose
    
    Returns:
        Formatted Text object ready for display
    """
    delta = halluc_forced - halluc_abstain
    
    result_text = Text()
    result_text.append("When model can say ", style="bold")
    result_text.append("'unknown'", style="cyan")
    result_text.append(": ", style="bold")
    result_text.append(display._format_percentage(halluc_abstain), style="cyan")
    result_text.append(" hallucination rate\n", style="dim")
    
    result_text.append("When model must pick ", style="bold")
    result_text.append("a weekday", style="magenta")
    result_text.append(": ", style="bold")
    result_text.append(display._format_percentage(halluc_forced), style="magenta")
    result_text.append(" hallucination rate\n\n", style="dim")
    
    result_text.append("Difference: ", style="bold")
    result_text.append(display.format_delta(delta))
    result_text.append(" (forcing an answer ", style="dim")
    if delta > 0:
        result_text.append("increases", style="red")
    elif delta < 0:
        result_text.append("decreases", style="green")
    else:
        result_text.append("doesn't change", style="yellow")
    result_text.append(" hallucination)", style="dim")
    
    return result_text


def print_architectural_effect_header() -> None:
    """Print header for architectural effect section."""
    display.print_newline()
    display.print_simple_text("[bold]ARCHITECTURAL EFFECT:[/bold]")
    display.print_simple_text_dim("Testing whether the model's output format affects hallucination rates.")
    display.print_newline()


def format_response_breakdown(
    n_trials: int,
    n_fabrications: int,
    n_abstentions: int,
    answers: List[str],
    confidences: List[float],
    raw_responses: List[str],
    confidence_threshold: float,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Format and display response breakdown statistics.
    
    Args:
        n_trials: Total number of trials
        n_fabrications: Number of fabrications (non-abstentions)
        n_abstentions: Number of abstentions
        answers: List of answer strings
        confidences: List of confidence values
        raw_responses: List of raw response strings
        confidence_threshold: Threshold for confidence
        verbosity: Verbosity level for display
    """
    if not should_show(verbosity):
        return
    
    # Verify classification math
    total_classified = n_fabrications + n_abstentions
    if total_classified != n_trials:
        display.print_simple_text(f"[red]ERROR: Classification mismatch: {n_fabrications} + {n_abstentions} = {total_classified} ≠ {n_trials}[/red]")
    
    hallucination_rate = n_fabrications / n_trials if n_trials > 0 else 0.0
    
    answer_counts = Counter(answers)
    
    # Show response breakdown with clear percentages
    fab_rate = n_fabrications / n_trials if n_trials > 0 else 0.0
    abs_rate = n_abstentions / n_trials if n_trials > 0 else 0.0
    fab_color = "red" if n_fabrications > 0 else "green"
    abs_color = "green" if n_abstentions > 0 else "yellow"
    
    display.print_simple_text(f"\n  [bold]Response Breakdown (N={n_trials} trials)[/bold]:")
    display.print_simple_text(f"    Fabrications: [{fab_color}]{n_fabrications}/{n_trials}[/{fab_color}] "
                             f"({display._format_percentage(fab_rate)}) - gave specific weekday answer")
    display.print_simple_text(f"    Abstentions: [{abs_color}]{n_abstentions}/{n_trials}[/{abs_color}] "
                             f"({display._format_percentage(abs_rate)}) - said 'unknown' or refused")
    display.print_simple_text(f"    [bold]Hallucination rate[/bold]: [magenta]{display._format_percentage(hallucination_rate)}[/magenta] "
                             f"(fabrications/total = {n_fabrications}/{n_trials})")
    
    # Classification verification
    if should_show(Verbosity.DEBUG):
        display.print_simple_text(f"\n    [dim]Classification check: {n_fabrications} + {n_abstentions} = {total_classified} ✓[/dim]")
    
    # Show actual responses
    if len(set(answers)) <= 3:
        answer_list = ", ".join(f"[cyan]{ans}[/cyan]" for ans in answers[:10])
        if len(answers) > 10:
            answer_list += f" [dim](+{len(answers)-10} more)[/dim]"
        display.print_simple_text(f"    Actual answers: {answer_list}")
    else:
        top_answers = answer_counts.most_common(3)
        answer_summary = ", ".join(f"[cyan]{ans}[/cyan]: {count}" for ans, count in top_answers)
        display.print_simple_text(f"    Distribution: {answer_summary}")
    
    # Show confidence stats
    if confidences:
        conf_stats = display._format_confidence_stats(confidences)
        # Extract just mean and range for simpler display
        mean_val = np.mean(confidences)
        min_val = min(confidences)
        max_val = max(confidences)
        display.print_simple_text(f"    Confidence: [cyan]{mean_val:.0%}[/cyan] mean, "
                                 f"range {min_val:.0%}-{max_val:.0%}")
    
    # Show raw responses only in debug mode
    if should_show(Verbosity.DEBUG):
        display.print_simple_text("\n    [dim]Individual responses:[/dim]")
        for i, raw_response in enumerate(raw_responses[:10]):
            status_color = "red" if confidences[i] > confidence_threshold else "green"
            display.print_simple_text(f"      [dim]#{i+1}: {raw_response} ({display._format_percentage(confidences[i])})[/dim]")
        if len(raw_responses) > 10:
            display.print_simple_text(f"      [dim]... +{len(raw_responses)-10} more responses[/dim]")


def format_hallucination_rate_summary(
    hallucination_rate: float,
    avg_confidence: float
) -> None:
    """
    Format and display hallucination rate summary.
    
    Args:
        hallucination_rate: Observed hallucination rate
        avg_confidence: Average confidence across trials
    """
    display.print_simple_text(f"\n  [bold green]✓[/bold green] [bold]Hallucination rate[/bold]: "
                 f"[magenta]{display._format_percentage(hallucination_rate, precision=1)}[/magenta] "
                 f"(avg confidence: [cyan]{display._format_percentage(avg_confidence, precision=1)}[/cyan])\n")


def format_context_measurement(
    context_label: str,
    answer: str,
    confidence: float,
    n_errors: int,
    prompt: str,
    context_name: str,
    dist: Dict[str, float],
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Format and display context measurement result.
    
    Args:
        context_label: Human-readable context label
        answer: Most common answer
        confidence: Confidence in the answer
        n_errors: Number of JSON validation errors
        prompt: Full prompt used (for debug mode)
        context_name: Context identifier
        dist: Answer distribution
        verbosity: Verbosity level for display
    """
    status_icon, status_color = display._status_icon_and_color(n_errors == 0)
    if n_errors > 0:
        status_icon = "⚠"
        status_color = "yellow"
    
    display.print_simple_text(f"  {status_icon} [bold]{context_label}[/bold] → [cyan]{answer}[/cyan]:[green]{confidence:.2f}[/green]" +
                 (f" [yellow]({n_errors} errors)[/yellow]" if n_errors > 0 else ""))
    
    # Show full prompt and distribution only on errors (debug mode)
    if n_errors > 0 and should_show(Verbosity.DEBUG):
        display.print_panel(prompt, title=f"[bold red]Context: {context_name} (ERRORS)[/bold red]", border_style="red", expand=False)
        dist_str = ", ".join(f"[cyan]{k}[/cyan]: [green]{v:.2f}[/green]" for k, v in dist.items() if v > 0.01)
        display.print_simple_text(f"  → Distribution: {dist_str}\n")


def format_output_format_comparison_table(
    halluc_rate_abstain: float,
    halluc_rate_forced: float,
    avg_conf_abstain: float,
    avg_conf_forced: float
) -> None:
    """
    Format and display output format comparison table.
    
    Args:
        halluc_rate_abstain: Hallucination rate when abstention allowed
        halluc_rate_forced: Hallucination rate when forced choice
        avg_conf_abstain: Average confidence when abstention allowed
        avg_conf_forced: Average confidence when forced choice
    """
    rate_delta = halluc_rate_forced - halluc_rate_abstain
    conf_delta = avg_conf_forced - avg_conf_abstain
    
    display.print_table(
        ["Metric", "Can Say 'Unknown'", "Must Choose Answer", "Difference"],
        [
            ["Hallucination rate", display._format_percentage(halluc_rate_abstain, precision=1), 
             display._format_percentage(halluc_rate_forced, precision=1), display.format_delta(rate_delta)],
            ["Average confidence", display._format_percentage(avg_conf_abstain, precision=1), 
             display._format_percentage(avg_conf_forced, precision=1), display.format_delta(conf_delta)],
        ],
        "Output Format Comparison"
    )


def format_task_contradiction_info(K: float, lower_bound: float) -> None:
    """
    Format and display task contradiction information.
    
    Args:
        K: Task contradiction in bits
        lower_bound: Theoretical minimum hallucination rate
    """
    display.print_simple_text(f"  [bold]Task contradiction[/bold] = [cyan]{K:.4f}[/cyan] bits, "
                 f"[bold]Theoretical minimum hallucination[/bold] = [yellow]{display._format_percentage(lower_bound, precision=2)}[/yellow]\n")


# ==============================================================================
# CALCULATION UTILITIES
# ==============================================================================

def calculate_decision_entropy(n_valid_outputs: int) -> float:
    """
    Calculate decision entropy H = log2(n_valid_outputs).

    Represents the architectural pressure independent of K.
    For 7 weekday options, H = 2.81 bits.
    """
    return math.log2(n_valid_outputs) if n_valid_outputs > 0 else 0.0


def calculate_witness_capacity(K: float, hallucination_rate: float) -> float:
    """
    Calculate implied witness capacity r from E + r ≥ K.

    If hallucination rate E and contradiction K are known,
    r ≤ K - E. Standard softmax has r ≈ 0 (no uncertainty channel).
    """
    return max(0.0, K - hallucination_rate)


def calculate_binomial_ci(n_successes: int, n_trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.

    Uses Wilson score interval which works better for small n than normal approximation.

    Args:
        n_successes: Number of successes
        n_trials: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n_trials == 0:
        return (0.0, 1.0)

    # Use 1.96 for 95% CI (standard normal z-score)
    z = 1.96 if confidence == 0.95 else math.sqrt(2) * math.erf(confidence / math.sqrt(2))

    p = n_successes / n_trials
    denominator = 1 + z**2 / n_trials
    center = (p + z**2 / (2 * n_trials)) / denominator
    margin = z * math.sqrt((p * (1 - p) / n_trials + z**2 / (4 * n_trials**2))) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Calculate binomial probability mass function."""
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def calculate_p_value_exceeds_bound(observed_rate: float, n_trials: int, theoretical_bound: float) -> float:
    """
    Calculate p-value for one-tailed binomial test of whether observed rate exceeds bound.

    H0: rate ≤ bound
    H1: rate > bound

    Args:
        observed_rate: Observed hallucination rate
        n_trials: Number of trials
        theoretical_bound: Theoretical lower bound

    Returns:
        P-value for exceeding bound
    """
    if n_trials == 0:
        return 1.0

    n_successes = int(observed_rate * n_trials)

    # Calculate exact binomial p-value: P(X >= k | p = bound)
    p_value = sum(binomial_pmf(k, n_trials, theoretical_bound)
                  for k in range(n_successes, n_trials + 1))

    return p_value


def calculate_cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for difference between two proportions.
    
    h = 2 * (arcsin(√p1) - arcsin(√p2))
    
    Interpretation:
    - |h| < 0.2: small effect
    - 0.2 ≤ |h| < 0.5: medium effect
    - |h| ≥ 0.5: large effect
    
    Args:
        p1: First proportion
        p2: Second proportion
    
    Returns:
        Cohen's h effect size
    """
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def calculate_brier_score(observed_rate: float, predicted_rate: float) -> float:
    """
    Calculate Brier score for calibration quality.
    
    Brier score = (observed - predicted)²
    Range: 0 (perfect) to 1 (worst)
    
    Args:
        observed_rate: Observed proportion
        predicted_rate: Predicted/probabilistic proportion
    
    Returns:
        Brier score
    """
    return (observed_rate - predicted_rate) ** 2


def calculate_witness_capacity_utilization(actual_r: float, theoretical_max_r: float) -> float:
    """
    Calculate witness capacity utilization percentage.
    
    utilization = (actual_r / theoretical_max_r) * 100
    
    Args:
        actual_r: Actual witness capacity used
        theoretical_max_r: Theoretical maximum witness capacity
    
    Returns:
        Utilization percentage (0-100)
    """
    if theoretical_max_r == 0:
        return 0.0 if actual_r == 0 else float('inf')
    return (actual_r / theoretical_max_r) * 100


def calculate_hellinger_distance(alpha_star: float) -> float:
    """
    Calculate Hellinger distance to frame-independence.
    
    D_H(P, FI) = √(1 - α*)
    
    Args:
        alpha_star: Agreement coefficient α*
    
    Returns:
        Hellinger distance
    """
    return math.sqrt(1 - alpha_star)


def calculate_tv_gap(K: float) -> float:
    """
    Calculate total variation gap bound.
    
    d_TV(P, FI) ≥ 1 - 2^(-K)
    
    Args:
        K: Contradiction in bits
    
    Returns:
        TV gap lower bound
    """
    return 1 - 2**(-K)


def format_response_classification_verification(
    answers: List[str],
    raw_responses: List[str],
    n_trials: int,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display response classification verification with explicit breakdown.
    
    Args:
        answers: List of answer strings
        raw_responses: List of raw response strings
        n_trials: Total number of trials
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    abstention_indicators = ['not specified', 'need to know', 'cannot', "don't know", 
                             'unable', 'unclear', 'insufficient', 'missing', 'unknown']
    
    weekday_answers = [ans for ans in answers if ans in weekdays]
    other_answers = [ans for ans in answers if ans not in weekdays]
    
    weekday_counts = Counter(weekday_answers)
    n_weekday = len(weekday_answers)
    n_unknown = len(other_answers)
    
    weekday_rate = n_weekday / n_trials if n_trials > 0 else 0.0
    unknown_rate = n_unknown / n_trials if n_trials > 0 else 0.0
    
    display.print_simple_text("\n  [bold]RESPONSE CLASSIFICATION VERIFICATION[/bold]:")
    display.print_simple_text(f"    Specific weekday answers: {n_weekday} ({display._format_percentage(weekday_rate)})")
    if weekday_counts:
        weekday_detail = ", ".join(f"{day}:{count}" for day, count in sorted(weekday_counts.items()))
        display.print_simple_text(f"      → {weekday_detail}")
    display.print_simple_text(f"    'Unknown' responses: {n_unknown} ({display._format_percentage(unknown_rate)})")
    display.print_simple_text(f"\n    Classification logic check:")
    display.print_simple_text(f"      {n_weekday} + {n_unknown} = {n_weekday + n_unknown} {'✓' if (n_weekday + n_unknown) == n_trials else '✗'} (accounts for all {n_trials} trials)")
    display.print_simple_text(f"\n    Hallucination rate = Fabrications / Total")
    display.print_simple_text(f"                      = {n_weekday} / {n_trials}")
    display.print_simple_text(f"                      = {display._format_percentage(weekday_rate)}")


def format_per_trial_breakdown(
    fabrications: List[Tuple[int, str, float]],
    abstentions: List[Tuple[int, str, float]],
    n_show: int = 5,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display per-trial breakdown table.
    
    Args:
        fabrications: List of (trial_num, response, confidence) tuples
        abstentions: List of (trial_num, response, confidence) tuples
        n_show: Number of trials to show
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    all_trials = sorted(fabrications + abstentions, key=lambda x: x[0])
    trials_to_show = all_trials[:n_show]
    
    if not trials_to_show:
        return
    
    rows = []
    for trial_num, response, confidence in trials_to_show:
        trial_type = "Fabrication" if (trial_num, response, confidence) in fabrications else "Abstention"
        correct = "✓" if trial_type == "Abstention" else "✗"
        rows.append([
            str(trial_num),
            response[:20] + ("..." if len(response) > 20 else ""),
            display._format_percentage(confidence),
            trial_type,
            correct
        ])
    
    display.print_table(
        ["Trial", "Response", "Confidence", "Type", "Correct?"],
        rows,
        f"Trial-by-Trial Results (showing first {len(trials_to_show)} of {len(all_trials)})"
    )


def format_confidence_analysis_by_type(
    fabrications: List[Tuple[int, str, float]],
    abstentions: List[Tuple[int, str, float]],
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display confidence analysis separated by response type.
    
    Args:
        fabrications: List of (trial_num, response, confidence) tuples
        abstentions: List of (trial_num, response, confidence) tuples
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    fab_confidences = [conf for _, _, conf in fabrications]
    abs_confidences = [conf for _, _, conf in abstentions]
    
    if not fab_confidences and not abs_confidences:
        return
    
    display.print_simple_text("\n  [bold]Confidence Analysis by Response Type:[/bold]")
    
    if fab_confidences:
        display.print_simple_text(f"    Fabrications: {display._format_confidence_stats(fab_confidences)}")
    
    if abs_confidences:
        display.print_simple_text(f"    Abstentions: {display._format_confidence_stats(abs_confidences)}")
    
    if fab_confidences and abs_confidences:
        fab_mean = np.mean(fab_confidences)
        abs_mean = np.mean(abs_confidences)
        if fab_mean > abs_mean:
            interpretation = "Model is MORE confident when fabricating"
            display.print_simple_text(f"    Interpretation: [yellow]{interpretation}[/yellow] (bad sign!)")
        else:
            interpretation = "Model is LESS confident when fabricating"
            display.print_simple_text(f"    Interpretation: [green]{interpretation}[/green]")


def format_witness_error_tradeoff(
    K: float,
    hallucination_rate: float,
    n_fabrications: int,
    n_abstentions: int,
    n_trials: int,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display witness-error tradeoff analysis (E + r ≥ K).
    
    Args:
        K: Task contradiction in bits
        hallucination_rate: Observed hallucination rate (E)
        n_fabrications: Number of fabrications
        n_abstentions: Number of abstentions
        n_trials: Total trials
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    E = hallucination_rate
    abstention_rate = n_abstentions / n_trials if n_trials > 0 else 0.0
    
    # Theoretical max r if all abstentions were used
    theoretical_max_r = -math.log2(abstention_rate) if abstention_rate > 0 else 0.0
    
    # Actual r (from witness capacity calculation)
    actual_r = max(0.0, K - E)
    
    display.print_simple_text("\n  [bold]Witness-Error Conservation Law (E + r ≥ K):[/bold]")
    display.print_simple_text(f"    Task contradiction: K = {K:.2f} bits")
    display.print_simple_text(f"    Fabrications: {n_fabrications}/{n_trials} ({display._format_percentage(E)}) → E = {E:.2f}")
    display.print_simple_text(f"    Abstentions: {n_abstentions}/{n_trials} ({display._format_percentage(abstention_rate)})")
    if abstention_rate > 0:
        display.print_simple_text(f"      → r = -log₂({abstention_rate:.2f}) ≈ {theoretical_max_r:.2f} bits (theoretical max)")
    display.print_simple_text(f"      → r ≈ {actual_r:.2f} bits (actual)")
    display.print_simple_text(f"    Check: E + r = {E:.2f} + {actual_r:.2f} = {E + actual_r:.2f} ≈ K = {K:.2f} {'✓' if abs(E + actual_r - K) < 0.1 else '✗'}")
    
    if theoretical_max_r > 0:
        utilization = calculate_witness_capacity_utilization(actual_r, theoretical_max_r)
        display.print_simple_text(f"    Utilization: {display._format_percentage(utilization / 100)} (model doesn't fully use witness channel)")


def format_decision_entropy_analysis(
    K: float,
    hallucination_rate: float,
    lower_bound: float,
    n_valid_outputs: int = 7,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display decision entropy contribution analysis.
    
    Args:
        K: Contradiction in bits
        hallucination_rate: Observed hallucination rate
        lower_bound: Theoretical lower bound from K
        n_valid_outputs: Number of valid output choices
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    H = calculate_decision_entropy(n_valid_outputs)
    total_pressure = K + H
    excess = hallucination_rate - lower_bound
    
    display.print_simple_text("\n  [bold]Sources of Hallucination Pressure:[/bold]")
    display.print_simple_text(f"    K (contradiction): {K:.2f} bits  ← Inevitable (from task structure)")
    display.print_simple_text(f"    H (choice entropy): {H:.2f} bits  ← log₂({n_valid_outputs} weekdays)")
    display.print_simple_text(f"    r (witness capacity): {max(0.0, K - hallucination_rate):.2f} bits  ← Actual uncertainty representation")
    display.print_simple_text(f"    Total pressure: K + H = {K:.2f} + {H:.2f} = {total_pressure:.2f} bits")
    display.print_simple_text(f"    Observed fabrication rate: {display._format_percentage(hallucination_rate)}")
    display.print_simple_text(f"    Expected from K alone: ≥{display._format_percentage(lower_bound)}")
    display.print_simple_text(f"    Excess: +{display._format_percentage(excess)} explained by H + low r")


def format_contrakit_bottleneck_analysis(
    worst_case_weights: Dict[Tuple[str, ...], float],
    context_scores: Dict[Tuple[str, ...], float],
    alpha_star: float,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display bottleneck analysis from contrakit.
    
    Args:
        worst_case_weights: λ* values from behavior.worst_case_weights
        context_scores: BC(p_c, q*) values from behavior.agreement.by_context().context_scores
        alpha_star: Agreement coefficient α*
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    display.print_simple_text("\n  [bold]Bottleneck Analysis (Contrakit Worst-Case Weights):[/bold]")
    
    rows = []
    for ctx_key, weight in worst_case_weights.items():
        ctx_name = ", ".join(ctx_key) if isinstance(ctx_key, tuple) else str(ctx_key)
        bc_value = context_scores.get(ctx_key, 0.0)
        is_bottleneck = weight > 1e-6
        is_active = abs(bc_value - alpha_star) < 1e-6
        
        bottleneck_status = "✓ Active" if is_bottleneck else "  Slack"
        bc_match = "Yes" if is_active else "No"
        
        rows.append([
            ctx_name,
            f"{bc_value:.4f}",
            bc_match,
            bottleneck_status
        ])
    
    display.print_table(
        ["Context", "BC(p_c,q*)", "α* match?", "Bottleneck?"],
        rows,
        "Context-Level Analysis"
    )
    
    active_contexts = [ctx for ctx, w in worst_case_weights.items() if w > 1e-6]
    if active_contexts:
        ctx_names = [", ".join(ctx) if isinstance(ctx, tuple) else str(ctx) for ctx in active_contexts]
        display.print_simple_text(f"    Interpretation: {', '.join(ctx_names)} are bottleneck constraints (λ* > 0)")
        display.print_simple_text("    These contexts prevent reconciliation with frame-independence")


def format_contrakit_geometric_analysis(
    alpha_star: float,
    K: float,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display geometric distance analysis from contrakit.
    
    Args:
        alpha_star: Agreement coefficient
        K: Contradiction in bits
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    hellinger_dist = calculate_hellinger_distance(alpha_star)
    tv_gap = calculate_tv_gap(K)
    
    display.print_simple_text("\n  [bold]Geometric Distance to Frame-Independence:[/bold]")
    display.print_simple_text(f"    Hellinger distance: D_H(P, FI) = √(1 - α*) = √(1 - {alpha_star:.4f}) = {hellinger_dist:.4f}")
    display.print_simple_text(f"    In 'contradiction space': P is {hellinger_dist:.4f} Hellinger units from FI")
    display.print_simple_text(f"    This is the minimum 'noise' needed to eliminate contradiction")
    display.print_simple_text(f"\n    Total Variation Gap:")
    display.print_simple_text(f"    Theory: d_TV(P, FI) ≥ 1 - 2^(-K) = {display._format_percentage(tv_gap)}")
    display.print_simple_text(f"    Interpretation: ANY frame-independent simulator must differ")
    display.print_simple_text(f"    from true behavior by at least {display._format_percentage(tv_gap)} on some context")


def format_theory_vs_reality_comparison(
    K: float,
    lower_bound: float,
    observed_rate: float,
    excess: float,
    E: float,
    r: float,
    has_bottlenecks: bool,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display theory vs reality comparison table.
    
    Args:
        K: Contradiction in bits
        lower_bound: Theoretical lower bound
        observed_rate: Observed hallucination rate
        excess: Excess beyond bound
        E: Error rate
        r: Witness capacity
        has_bottlenecks: Whether bottleneck contexts exist
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    predictions = [
        ("K > 0 forces halluc", observed_rate > 0, f"{display._format_percentage(observed_rate)} > 0%"),
        (f"Rate ≥ 1-2^(-K) = {display._format_percentage(lower_bound)}", observed_rate >= lower_bound, 
         f"{display._format_percentage(observed_rate)} ≥ {display._format_percentage(lower_bound)}"),
        ("Excess from H + low r", excess > 0, f"+{display._format_percentage(excess)}"),
        ("Bottlenecks = active", has_bottlenecks, "Yes" if has_bottlenecks else "No"),
        (f"E + r ≥ K", abs(E + r - K) < 0.1, f"{E:.2f}+{r:.2f}≈{K:.2f}")
    ]
    
    rows = []
    for pred, observed, match_str in predictions:
        match_icon = "✓ Yes" if observed else "✗ No"
        rows.append([pred, match_str, match_icon])
    
    display.print_table(
        ["Prediction (from theory)", "Observed", "Match?"],
        rows,
        "THEORY VS OBSERVATION"
    )


def format_statistical_significance(
    observed_rate: float,
    n_trials: int,
    lower_bound: float,
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display statistical significance testing results.
    
    Args:
        observed_rate: Observed hallucination rate
        n_trials: Number of trials
        lower_bound: Theoretical lower bound
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    p_value = calculate_p_value_exceeds_bound(observed_rate, n_trials, lower_bound)
    ci_lower, ci_upper = calculate_binomial_ci(int(observed_rate * n_trials), n_trials)
    cohens_h = calculate_cohens_h(observed_rate, lower_bound)
    
    display.print_simple_text("\n  [bold]Statistical Tests:[/bold]")
    display.print_simple_text(f"    Hallucination exceeds bound ({display._format_percentage(observed_rate)} > {display._format_percentage(lower_bound)}):")
    
    # Invert threshold logic: lower p-value is better (green), higher is worse (red)
    p_color = "green" if p_value < 0.05 else "yellow" if p_value < 0.10 else "red"
    p_significance = "highly significant" if p_value < 0.01 else "significant" if p_value < 0.05 else "marginally significant" if p_value < 0.10 else "not significant"
    display.print_simple_text(f"      Binomial test: p = [{p_color}]{p_value:.3f}[/{p_color}] ({p_significance})")
    
    effect_size_label = "large" if abs(cohens_h) >= 0.5 else "medium" if abs(cohens_h) >= 0.2 else "small"
    display.print_simple_text(f"      Effect size: Cohen's h = {cohens_h:.2f} ({effect_size_label} effect)")
    display.print_simple_text(f"      95% CI: [{display._format_percentage(ci_lower)}, {display._format_percentage(ci_upper)}]")


def format_confidence_histogram(
    fabrications: List[Tuple[int, str, float]],
    abstentions: List[Tuple[int, str, float]],
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display confidence histogram with bins.
    
    Args:
        fabrications: List of (trial_num, response, confidence) tuples
        abstentions: List of (trial_num, response, confidence) tuples
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    fab_confidences = [conf for _, _, conf in fabrications]
    abs_confidences = [conf for _, _, conf in abstentions]
    
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    bin_labels = ["0-50%", "50-70%", "70-90%", "90-100%"]
    
    fab_counts = [sum(1 for c in fab_confidences if low <= c < high) for low, high in bins]
    abs_counts = [sum(1 for c in abs_confidences if low <= c < high) for low, high in bins]
    
    display.print_simple_text("\n  [bold]Confidence Distribution:[/bold]")
    rows = []
    for i, (label, fab_count, abs_count) in enumerate(zip(bin_labels, fab_counts, abs_counts)):
        total = fab_count + abs_count
        fab_pct = (fab_count / total * 100) if total > 0 else 0
        abs_pct = (abs_count / total * 100) if total > 0 else 0
        rows.append([
            label,
            f"{fab_count} ([red]{fab_pct:.0f}%[/red])",
            f"{abs_count} ([blue]{abs_pct:.0f}%[/blue])",
            str(total)
        ])
    
    display.print_table(
        ["Confidence", "Fabrications", "Abstentions", "Total"],
        rows,
        "Confidence Histogram (N={})".format(len(fab_confidences) + len(abs_confidences))
    )
    
    if fab_confidences:
        high_conf_fab = sum(1 for c in fab_confidences if c >= 0.9)
        display.print_simple_text(f"    Interpretation: {high_conf_fab}/{len(fab_confidences)} fabrications ({high_conf_fab/len(fab_confidences):.0%}) have 90%+ confidence")
        display.print_simple_text(f"    → Model is overconfident when fabricating")


def format_confidence_calibration(
    fabrications: List[Tuple[int, str, float]],
    abstentions: List[Tuple[int, str, float]],
    verbosity: int = Verbosity.NORMAL
) -> None:
    """
    Display confidence calibration analysis.
    
    Args:
        fabrications: List of (trial_num, response, confidence) tuples
        abstentions: List of (trial_num, response, confidence) tuples
        verbosity: Verbosity level
    """
    if not should_show(verbosity):
        return
    
    all_responses = fabrications + abstentions
    if not all_responses:
        return
    
    bins = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.0, 0.7)]
    bin_labels = ["90-100%", "80-90%", "70-80%", "<70%"]
    
    rows = []
    total_brier = 0.0
    for i, ((low, high), label) in enumerate(zip(bins, bin_labels)):
        bin_responses = [(t, r, c) for t, r, c in all_responses if low <= c < high]
        count = len(bin_responses)
        
        # Accuracy = 0% for fabrications (all wrong), 100% for abstentions (all correct)
        correct = sum(1 for t, r, c in bin_responses if (t, r, c) in abstentions)
        accuracy = (correct / count * 100) if count > 0 else 0.0
        
        # Calibration error = |confidence - accuracy|
        avg_confidence = np.mean([c for _, _, c in bin_responses]) if bin_responses else 0.0
        calibration_error = abs(avg_confidence * 100 - accuracy)
        
        # Brier score contribution
        brier = calculate_brier_score(accuracy / 100, avg_confidence)
        total_brier += brier * count
        
        rows.append([
            label,
            str(count),
            f"{correct} ({accuracy:.0%})",
            f"{calibration_error:.0%} ({'over' if avg_confidence * 100 > accuracy else 'under'})"
        ])
    
    total_brier /= len(all_responses)
    
    display.print_simple_text("\n  [bold]Confidence Calibration:[/bold]")
    display.print_table(
        ["Confidence", "Count", "Correct", "Calibration Error"],
        rows,
        "Calibration Analysis"
    )
    display.print_simple_text(f"    Brier Score: {total_brier:.3f} (0=perfect, 1=worst)")
    display.print_simple_text(f"    Expected: High confidence → high accuracy")
    display.print_simple_text(f"    Observed: High confidence → low accuracy (overconfident)")
