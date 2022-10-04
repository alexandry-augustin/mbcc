using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using utils;
using System;
using MicrosoftResearch.Infer.Collections;

namespace utils.mbcc
{
	/// <summary>
	/// This class represents a single datum for distribution-based models.
	/// </summary>
    public class DatumDist
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
        /// The worker array
        /// </summary>
        public int[] WorkerArray;
        /// <summary>
        /// The worker's distribution
        /// </summary>
        public Vector WorkerDistr;
        /// <summary>
        /// The array of gold standard values. It is null if not available.
        /// </summary>
        public int?[] GoldArr;
        /// <summary>
        /// The gold standard distribution.
        /// </summary>
        public Vector GoldDistr;
	}
}
