using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;

namespace utils
{
    /// <summary>
    /// This class represents a single datum for single-label-based models.
    /// </summary>
    [Serializable()]
    public class Datum
    {
        /// <summary>
        /// The worker id.
        /// </summary>
        public string WorkerId;
        /// <summary>
        /// The task id.
        /// </summary>
        public string TaskId;
        /// <summary>
        /// The worker's label.
        /// </summary>
        public int WorkerLabel;
		/// <summary>
		/// The worker's probability.
		/// </summary>
		public double WorkerProb;
        /// <summary>
        /// The task's gold label (optional).
        /// </summary>
        public int? GoldLabel;
		/// <summary>
		/// The task's gold probability
		/// </summary>
		public double GoldProb;
	}
}