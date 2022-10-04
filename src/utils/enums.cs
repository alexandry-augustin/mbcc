namespace utils
{
    /// <summary>
    /// Options for which model to run.
    /// </summary>
    public enum RunType
    {
        /// <summary>
        /// The true label distribution
        /// as given by the normalised workers' label counts.
        /// </summary>
        LinOp = 0,
        /// <summary>
        /// The BCC model.
        /// </summary>
        BCC = 1,
		/// <summary>
		/// The MBCC model.
		/// </summary>
		MBCC = 2,
		/// <summary>
		/// Uniform (reference).
		/// </summary>
		UNIFORM = 3,
		/// <summary>
		/// Median of the judgments.
		/// </summary>
		MEDIAN = 4,
    }
	/// <summary>
	/// Options for which model to run.
	/// </summary>
	public enum ErrorType
	{
		/// <summary>
		/// KL-divergence
		/// </summary>
		KLDivergence=0,
		/// <summary>
		/// Eucledian distance
		/// </summary>
		EucledianDistance=1,
		/// <summary>
		/// Mean squared error
		/// </summary>
		MeanSquaredError=2
	}
	/// <summary>
	/// Spamming strategy.
	/// </summary>
	public enum SpammingStrategy
	{
		/// <summary>
		/// Single label.
		/// </summary>
		SingleLabel=0,
		/// <summary>
		/// Uniform.
		/// </summary>
		Uniform=1,
		/// <summary>
		/// Uniform with prior.
		/// </summary>
		UniformWithPrior=2,
		/// <summary>
		/// Perfect.
		/// </summary>
		Perfect=3,
		/// <summary>
		/// Perfect with prior.
		/// </summary>
		PerfectWithPrior=4,
		/// <summary>
		/// Spares Uniform with prior.
		/// </summary>
		SparesUniformWithPrior=5,
	}
}