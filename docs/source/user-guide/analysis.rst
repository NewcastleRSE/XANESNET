Analysis configuration
======================

The analysis workflow reads saved predictions from :doc:`inference`.
It selects samples, computes per-sample values, reduces them to summaries, and writes reports and figures.
The pipeline is independent of model training and inference.

An analysis configuration requires five lists:

* ``selectors`` choose samples or sample groups.
* ``collectors`` compute values for each selected sample.
* ``aggregators`` reduce collected values to summaries.
* ``reporters`` write structured results.
* ``plotters`` write figures and table outputs.

Analysis pipeline
-----------------

Analysis starts with one prediction reader for each inference run.
The readers provide prediction records and, when the inference datasource can be loaded, matching structures.

1. Selectors create the sample streams to analyse. Each configured selector is applied independently to each prediction reader.
   Use a ``chain`` selector when several filters must be applied in sequence.
2. Collectors process every sample in each selected stream. Their per-sample values are written to JSONL files under the analysis run's ``aux/`` directory.
3. Aggregators read the selector streams and collector values and reduce them to summary results such as errors, distributions, rankings, or per-channel statistics.
4. Reporters and plotters consume the combined analysis results. Reporters write structured files, while plotters create figures and tables.

With this pipeline it is easily possible to analyze and compare multiple inference runs.

Available implementations
--------------------------

Selectors
~~~~~~~~~

* ``all`` and ``none`` (:class:`~xanesnet.analysis.selectors.identity.IdentitySelector`) keep all samples.
* ``index_list`` (:class:`~xanesnet.analysis.selectors.by_index.IndexSelector`) selects explicit zero-based indices.
* ``index_range`` (:class:`~xanesnet.analysis.selectors.by_range.IndexRangeSelector`) selects an inclusive ``start`` to exclusive ``end`` range.
* ``random`` (:class:`~xanesnet.analysis.selectors.bernoulli.BernoulliSelector`) keeps samples with probability ``p``.
* ``element`` (:class:`~xanesnet.analysis.selectors.by_element.ElementSelector`) selects target sites by chemical element.
* ``structure_cluster`` (:class:`~xanesnet.analysis.selectors.structure_cluster.StructureClusterSelector`) partitions structures using a descriptor.
* ``chain`` (:class:`~xanesnet.analysis.selectors.chain.ChainSelector`) applies an ordered sequence of selectors.

Collectors
~~~~~~~~~~

* ``loss`` (:class:`~xanesnet.analysis.collectors.loss.LossCollector`) computes a configured loss for each sample and can collect an energy-resolved vector.
* ``descriptor`` (:class:`~xanesnet.analysis.collectors.descriptor.DescriptorCollector`) computes a descriptor for each matched structure.

Aggregators
~~~~~~~~~~~

* ``scalar`` (:class:`~xanesnet.analysis.aggregators.scalar.ScalarAggregator`) computes scalar summary statistics.
* ``vector`` (:class:`~xanesnet.analysis.aggregators.vector.VectorAggregator`) summarises vector values such as energy-resolved losses.
* ``spectrum`` (:class:`~xanesnet.analysis.aggregators.spectrum.SpectrumAggregator`) computes per-channel prediction and target statistics.
* ``ranking`` (:class:`~xanesnet.analysis.aggregators.ranking.RankingAggregator`) selects best and worst samples using a collected value.
* ``bias_variance`` (:class:`~xanesnet.analysis.aggregators.bias_variance.BiasVarianceAggregator`) computes a per-channel bias and variance decomposition.

Reporters
~~~~~~~~~

* ``scalar`` (:class:`~xanesnet.analysis.reporters.scalar.ScalarReporter`) writes per-sample scalar values as CSV files.
* ``statistics`` (:class:`~xanesnet.analysis.reporters.statistics.StatisticsReporter`) writes aggregated values as YAML or JSON.

Plotters
~~~~~~~~

* ``scalar`` (:class:`~xanesnet.analysis.plotters.scalar.ScalarPlotter`) plots scalar-value distributions.
* ``spectra_all`` (:class:`~xanesnet.analysis.plotters.spectra.AllSpectraPlotter`) plots selected prediction and target spectra.
* ``spectra_comparison`` (:class:`~xanesnet.analysis.plotters.spectra_comparison.SpectraComparisonPlotter`) compares best and worst samples using ``sort_key``.
* ``stat_table`` (:class:`~xanesnet.analysis.plotters.stat_table.StatTablePlotter`) writes PDF tables of aggregated statistics.
* ``stat_table_latex`` (:class:`~xanesnet.analysis.plotters.stat_table_latex.StatTableLatexPlotter`) writes LaTeX sources and optional PDFs.
* ``energy_resolved_loss`` (:class:`~xanesnet.analysis.plotters.energy_resolved.EnergyResolvedLossPlotter`) plots energy-resolved loss curves.
* ``mean_spectrum`` (:class:`~xanesnet.analysis.plotters.mean_spectrum.MeanSpectrumPlotter`) plots mean spectra and best or worst tail means.
* ``parity`` (:class:`~xanesnet.analysis.plotters.parity.ParityPlotter`) plots predicted-versus-target intensity parity.
* ``bias_variance`` (:class:`~xanesnet.analysis.plotters.bias_variance.BiasVariancePlotter`) plots per-channel bias and variance.
* ``error_correlation`` (:class:`~xanesnet.analysis.plotters.error_correlation.ErrorCorrelationPlotter`) plots pairwise error correlations using ``sort_key``.
* ``pca`` (:class:`~xanesnet.analysis.plotters.pca.PcaPlotter`) plots structure groups in descriptor PCA space.
* ``selector_overview`` (:class:`~xanesnet.analysis.plotters.selector_overview.SelectorOverviewPlotter`) compares selector groups using ``err_key``.

Common configuration
--------------------

The optional top-level fields are ``seed`` and ``preload``.
``preload`` loads prediction records and matched structures during setup instead of reading them on demand.
This can speed up analysis significantly for small test datasets.

Examples
--------

Independent selector streams:

.. code-block:: yaml

   selectors:
     - selector_type: element
       elements: [Fe]
     - selector_type: random
       p: 0.1

Sequential filtering:

.. code-block:: yaml

   selectors:
     - selector_type: chain
       chain:
         - selector_type: element
           elements: [Fe]
         - selector_type: random
           p: 0.1

A complete combination of analysis components is shown in ``configs/analyze_example.yaml``.

API reference
-------------

See also :mod:`xanesnet.analysis` and the schemas under ``xanesnet/schemas/analysis/`` for complete fields and defaults.
