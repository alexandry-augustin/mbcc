using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using utils;

namespace utils.mbcc
{
    class DataMappingMultiClass
    {
		/// <summary>
		/// The mapping from the worker index to the worker id.
		/// </summary>
		public string[] WorkerIndexToId;
		/// <summary>
		/// The mapping from the worker id to the worker index.
		/// </summary>
		public Dictionary<string, int> WorkerIdToIndex;
		/// <summary>
		/// The mapping from the task index to the task id.
		/// </summary>
		public string[] TaskIndexToId;
		/// <summary>
		/// The mapping from the task id to the task index.
		/// </summary>
		public Dictionary<string, int> TaskIdToIndex;
		/// <summary>
		/// The lower bound of the labels range.
		/// </summary>
		public int LabelMin;
		/// <summary>
		/// The upper bound of the labels range.
		/// </summary>
		public int LabelMax;
		/// <summary>
		/// The number of label values.
		/// </summary>
		public int LabelCount
		{
			get
			{
				return LabelMax - LabelMin + 1;
			}
		}
		/// <summary>
		/// The number of workers.
		/// </summary>
		public int WorkerCount
		{
			get
			{
				return WorkerIndexToId.Length;
			}
		}
		/// <summary>
		/// The number of tasks.
		/// </summary>
		public int TaskCount
		{
			get
			{
				return TaskIndexToId.Length;
			}
		}
        /// <summary>
        /// The enumerable list of multi class data.
        /// </summary>
		public new  IList<DatumDist> Data
        {
            get;
            private set;
        }
        /// <summary>
        /// The enumerable list of multi class data.
        /// </summary>
        public new IEnumerable<DatumDist> DataWithGold
        {
            get;
            private set;
        }
        /// <summary>
        /// Creates a data mapping.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="labelMin">The lower bound of the labels range.</param>
        /// <param name="labelMax">The upper bound of the labels range.</param>
		public DataMappingMultiClass(IList<DatumDist> data, int labelMin = int.MaxValue, int labelMax = int.MinValue)
        {
			this.WorkerIndexToId = data.Select(d => d.WorkerId).Distinct().ToArray();
			this.WorkerIdToIndex = WorkerIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
			this.TaskIndexToId = data.Select(d => d.TaskId).Distinct().ToArray();
			this.TaskIdToIndex = TaskIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
			int numLabels = 0;
			if(data.Count>0)
			{
	            //numLabels = data.First().WorkerArray.Length;
				numLabels = data.First().WorkerDistr.Count;
			}

            if (labelMin <= labelMax)
            {
                this.LabelMin = labelMin;
				this.LabelMax = labelMax;
            }
            else
            {
				this.LabelMin = 0;
				this.LabelMax = numLabels-1;
            }
            this.Data = data;
            this.DataWithGold = data.Where(d => d.GoldDistr != null);
        }
        /// <summary>
        /// Returns the the gold distribution of each task.
        /// </summary>
        /// <returns>The dictionary keyed by task id and the value is the gold distribution.</returns>
        public Dictionary<string, Vector> GetGoldDistrPerTaskId()
        {
            // Gold distribution that are not consistent are returned as null
            // Distribution are returned as indexed by task index
            return Data.GroupBy(d => d.TaskId).
              Select(t => t.GroupBy(d => d.GoldDistr).Where(d => d.Key != null)).
              Where(gold_d => gold_d.Count() > 0).
              Select(gold_d =>
              {
                  int count = gold_d.Distinct().Count();
                  var datum = gold_d.First().First();
                  if (count == 1)
                  {
                      var gold = datum.GoldDistr;
                      return new Tuple<string, Vector>(datum.TaskId, gold);
                  }
                  else
                  {
                      return new Tuple<string, Vector>(datum.TaskId, null);
                  }
              }).ToDictionary(tup => tup.Item1, tup => tup.Item2);
        }
		/// <summary>
		/// Samples workers' distributions
		/// </summary>
		/// <returns>The multi class data.</returns>
		/// <param name="data">Data.</param>
		/// <param name="numSamplesPerWorker">Number samples per worker.</param>
        public List<Datum> SampleMultiClassData(IList<DatumDist> data, int numSamplesPerWorker)
        {
			Console.WriteLine("---- Sampling Distributions ----");
			DateTime start = DateTime.Now;
            
			List<Datum> sampledData = new List<Datum>();
			foreach (DatumDist d in data)
            {
				Datum[] currWorkerSamples = Util.ArrayInit(numSamplesPerWorker, i =>
                    new Datum()
                    {
                        WorkerId = d.WorkerId,
                        TaskId = d.TaskId,
                        WorkerLabel = Discrete.Sample(d.WorkerDistr)
                    });
				sampledData = sampledData.Concat(currWorkerSamples).ToList();
            }

			Console.WriteLine("---- End Sampling ----");

			TimeSpan duration = DateTime.Now-start;
			Console.WriteLine ("Sampling duration ({0} samples per distribution): {1} seconds", numSamplesPerWorker, duration.TotalSeconds);

            return sampledData;
        }

//FIXME: duplicates from DataMapping


		/// <summary>
		/// Returns the matrix of the task indices (columns) of each worker (rows).
		/// </summary>
		/// <param name="data">The data.</param>
		/// <returns>The matrix of the task indices (columns) of each worker (rows).</returns>
		public int[][] GetTaskIndicesPerWorkerIndex(IEnumerable<Datum> data)
		{
			int[][] result = new int[WorkerCount][];
			for (int i = 0; i < WorkerCount; i++)
			{
				var wid = WorkerIndexToId[i];
				result[i] = data.Where(d => d.WorkerId == wid).Select(d => TaskIdToIndex[d.TaskId]).ToArray();
			}

			return result;
		}
		/// <summary>
		/// Returns the matrix of the labels (columns) of each worker (rows).
		/// </summary>
		/// <param name="data">The data.</param>
		/// <returns>The matrix of the labels (columns) of each worker (rows).</returns>
		public int[][] GetLabelsPerWorkerIndex(IEnumerable<Datum> data)
		{
			int[][] result = new int[WorkerCount][];
			for (int i = 0; i < WorkerCount; i++)
			{
				var wid = WorkerIndexToId[i];
				result[i] = data.Where(d => d.WorkerId == wid).Select(d => d.WorkerLabel - LabelMin).ToArray();
			}

			return result;
		}
    }
}
