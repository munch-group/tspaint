"""The read-out surface shared by every per-haplotype, per-locus soft posterior.

Both deliverables in tspaint are the same *shape* of object — a dict mapping each haplotype to a
list of contiguous :class:`~tspaint.output.Segment`\\ s that tile ``[0, L)``, each carrying a soft
``(K,)`` posterior — even though they come from very different models:

* :class:`tspaint.Painting` — ancestry from the CTMC blocked-EM fit (``posterior`` = over ancestry
  states), plus an ancestry-specific tail (``Q``/``π``/``w``, :meth:`~tspaint.Painting.rate_through_time`,
  :meth:`~tspaint.Painting.introgression_map`);
* :class:`tspaint.GhostResult` — ``P(ghost)`` from the depth-emission HMM (``posterior`` =
  ``[P(modern), P(ghost)]``), plus a ghost-specific tail (``μ``/``σ``/``A``/``π₀``, ``burden``,
  :meth:`~tspaint.GhostResult.tracts`).

:class:`SoftTrack` factors out the model-agnostic read-out — :meth:`segments` (hard tracts with the
calibrated dead-band), :meth:`posterior_at`, and :meth:`plot` — so both classes share one
implementation instead of each re-deriving it. It is a behaviour mixin, not a dataclass: a subclass
is the dataclass and supplies the data attributes the mixin reads (see :class:`SoftTrack`).
"""
from __future__ import annotations

from .output import hard_segments, posterior_at

__all__ = ["SoftTrack"]


class SoftTrack:
    """Mixin providing the shared read-out surface for a per-locus soft posterior along the genome.

    A subclass must be a :func:`~dataclasses.dataclass` that provides:

    * ``posteriors`` — ``dict[int, list[Segment | MergedSegment]]`` tiling ``[0, L)`` per haplotype
      (e.g. from :func:`tspaint.output.posterior_table` or
      :func:`tspaint.ensemble.merge_posterior_tables`);
    * ``default_deadband`` — :class:`float`, the default dead-band for :meth:`segments`;
    * ``_seqlen`` — the genome length (or override :attr:`length`).

    A subclass may set the class attributes ``_hi_state`` / ``_hi_label`` to control which state's
    probability drives the :meth:`plot` colour scale and how the colour bar is labelled (the
    "highlighted" state, drawn at the top of the diverging colormap). Defaults highlight state 0 and
    label it ``"P(ancestry A)"`` (so :class:`tspaint.Painting` needs no override).
    """

    # Which state the plot highlights (top of the diverging colormap) and its colour-bar label.
    # Subclasses override (e.g. GhostResult uses state 1 = ghost, "P(ghost)").
    _hi_state = 0
    _hi_label = "P(ancestry A)"

    @property
    def length(self):
        """Sequence length of the painted genome (default: the stored ``_seqlen``)."""
        return self._seqlen

    @property
    def samples(self):
        """Ordered haplotype ids carried by this track — the row order used by :meth:`plot`."""
        return list(self.posteriors)

    def segments(self, deadband=None):
        """Hard ancestry segments for downstream tract-length / dating analysis.

        Parameters
        ----------
        deadband : float, optional
            Confidence dead-band suppressing low-confidence flips that fragment long
            tracts. Defaults to :attr:`default_deadband`. See
            :func:`tspaint.output.hard_segments`.

        Returns
        -------
        dict[int, list[tuple[float, float, int]]]
            Per haplotype, hard ``(left, right, state)`` segments.
        """
        db = self.default_deadband if deadband is None else deadband
        return {q: hard_segments(t, db) for q, t in self.posteriors.items()}

    def posterior_at(self, sample, position):
        """Posterior vector for ``sample`` at a genomic ``position``.

        Parameters
        ----------
        sample : int
            Sample-node id to look up.
        position : float
            Genomic position.

        Returns
        -------
        numpy.ndarray or None
            The ``(K,)`` posterior covering ``position``, or ``None`` if uncovered.
        """
        return posterior_at(self.posteriors, sample, position)

    def plot(self, truth=None, title=None, cmap='coolwarm', colors=None, return_plot=False):
        """Stacked strip plot of the soft posterior — one row per haplotype.

        Each haplotype row shows, top to bottom: the **soft** per-position posterior for the
        highlighted state (``self._hi_label``, e.g. ``P(ancestry A)`` for a painting or ``P(ghost)``
        for the ghost detector) as a colour gradient, the **hard** segments (:meth:`segments` at
        ``deadband=0.4``), and — when ``truth`` is given — the true tracts as a reference track
        beneath. A shared colour bar maps the highlighted state's probability (highlighted state at
        the top of ``cmap``, the other at the bottom). For an ensemble result the rows show the
        ensemble-mean posterior. Requires matplotlib.

        Parameters
        ----------
        truth : dict[int, list[tuple[float, float, int]]], optional
            Ground-truth tracts ``(left, right, state)`` per haplotype, drawn as a reference track
            below each row. No truth track is drawn when ``None`` (default).
        title : str, optional
            Title placed on the top haplotype axes.
        cmap : str or matplotlib.colors.Colormap, optional
            Diverging colormap mapping the highlighted state's probability ``∈ [0, 1]`` (highlighted
            state → ``1.0``, the other → ``0.0``). Default ``'coolwarm'``. Ignored when ``colors`` is
            given.
        colors : list, optional
            Colours to build a custom diverging colormap from, overriding ``cmap`` — e.g.
            ``["#2c7bb6", "#ffffbf", "#d7191c"]``.
        return_plot : bool, optional
            Return the Matplotlib ``(figure, axes)`` so the caller can further customise or save
            the plot, instead of ``None``. Default ``False``.

        Returns
        -------
        tuple or None
            ``(figure, list_of_axes)`` if ``return_plot`` is ``True``, else ``None``.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        hi = self._hi_state
        qs = self.samples
        segments = self.segments(deadband=0.4)

        if colors:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)

        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap)
        fig = plt.figure(figsize=(9, 0.3 * len(qs) + 1))
        gs = fig.add_gridspec(len(qs), 2, width_ratios=[1, 0.03], hspace=0)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(qs))]

        for i, q in enumerate(qs):
            ymin, ymax = 0, 1.5
            if truth:
                ymin = -0.25
                for (l, r, s) in truth[q]:
                    axes[i].barh(-0.25, r - l, left=l, height=0.5,
                                 color=sm.to_rgba(1.0 if s == hi else 0.0), edgecolor="none")
            for (l, r, s) in segments[q]:
                axes[i].barh(0.25, r - l, left=l, height=0.5,
                             color=sm.to_rgba(1.0 if s == hi else 0.0), edgecolor="none")
            axes[i].axhline(0.25, c='black', lw=0.25)
            axes[i].axhline(0.5, c='black', lw=0.25)
            for seg in self.posteriors[q]:
                axes[i].barh(1, seg.right - seg.left, left=seg.left, height=1,
                             color=sm.to_rgba(seg.posterior[hi]), edgecolor="none")

            axes[i].set_ylim(ymin, ymax)
            axes[i].set_xlim(0, self.length)
            axes[i].set_ylabel(f'hapl. {i}', rotation=0, fontsize=7, color="0.0", horizontalalignment="right")
            axes[i].yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            if i < len(axes) - 1:
                axes[i].xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            axes[i].tick_params(axis='x', bottom=True)
            axes[i].spines['top'].set_visible(True)
            axes[i].spines['bottom'].set_visible(True)
            axes[i].spines['left'].set_visible(True)
            axes[i].spines['right'].set_visible(True)
            axes[i].grid(False)

        ax = fig.add_subplot(gs[:, 1])
        ax.set_axis_off()
        cb = fig.colorbar(sm, ax=ax, fraction=0.5, pad=0.01)
        cb.set_label(self._hi_label)
        if title:
            axes[0].set_title(title)
        plt.tight_layout()
        if return_plot:
            return fig, axes
