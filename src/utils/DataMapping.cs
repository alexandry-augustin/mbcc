using System;
using System.Collections.Generic;
using System.Linq;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;

namespace utils
{
    /// <summary>
    /// Data mapping class. This class manages the mapping between the data (which is
    /// in the form of task, worker ids, and labels) and the model data (which is in term of indices).
    /// </summary>
    public class DataMapping
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
        /// The enumerable list of data.
        /// </summary>
        public IEnumerable<Datum> Data
        {
            get;
            private set;
        }
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
		/// Constructor
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="labelMin">The lower bound of the labels range.</param>
        /// <param name="labelMax">The upper bound of the labels range.</param>
        public DataMapping(IEnumerable<Datum> data, int labelMin = int.MaxValue, int labelMax = int.MinValue)
		//public DataMapping(IEnumerable<Datum> data, int labelMin, int labelMax)
        {
			this.WorkerIndexToId = data.Select(d => d.WorkerId).Distinct().ToArray(); //array of unique WorkerId
			this.WorkerIdToIndex = WorkerIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
			this.TaskIndexToId = data.Select(d => d.TaskId).Distinct().ToArray();
			this.TaskIdToIndex = TaskIndexToId.Select((id, idx) => new KeyValuePair<string, int>(id, idx)).ToDictionary(x => x.Key, y => y.Value);
			int[] labels = data.Select(d => d.WorkerLabel).Distinct().OrderBy(lab => lab).ToArray();

            if (labelMin <= labelMax)
            {
				this.LabelMin = labelMin;
				this.LabelMax = labelMax;
            }
            else
            {
				this.LabelMin = labels.Min();
				this.LabelMax = labels.Max();
            }
            this.Data = data;
        }
        /// <summary>
        /// Returns the matrix of the task indices (columns) of each worker (rows).
        /// </summary>
        /// <param name="data">The data.</param>
        /// <returns>The matrix of the task indices (columns) of each worker (rows).</returns>
        public int[][] GetTaskIndicesPerWorkerIndex(IEnumerable<Datum> data)
        {
            int[][] result = new int[this.WorkerCount][];
			for (int i = 0; i < this.WorkerCount; i++)
            {
                string wid = WorkerIndexToId[i];
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
			int[][] result = new int[this.WorkerCount][];
			for (int i = 0; i < this.WorkerCount; i++)
            {
                string wid = WorkerIndexToId[i];
                result[i] = data.Where(d => d.WorkerId == wid).Select(d => d.WorkerLabel - LabelMin).ToArray();
            }

            return result;
        }
        /// <summary>
        /// Returns the the gold labels of each task.
        /// </summary>
        /// <returns>The dictionary keyed by task id and the value is the gold label.</returns>
        public Dictionary<string, int?> GetGoldLabelsPerTaskId()
        {
            // Gold labels that are not consistent are returned as null
            // Labels are returned as indexed by task index
            return Data.GroupBy(d => d.TaskId).
              Select(t => t.GroupBy(d => d.GoldLabel).Where(d => d.Key != null)).
              Where(gold_d => gold_d.Count() > 0).
              Select(gold_d =>
              {
                  int count = gold_d.Distinct().Count();
                  Datum datum = gold_d.First().First();
                  if (count == 1)
                  {
                      int? gold = datum.GoldLabel;
                      if (gold != null)
                          gold = gold.Value - LabelMin;
                      return new Tuple<string, int?>(datum.TaskId, gold);
                  }
                  else
                  {
                      return new Tuple<string, int?>(datum.TaskId, (int?)null);
                  }
              }).ToDictionary(tup => tup.Item1, tup => tup.Item2);
        }
		/// <summary>
		/// Returns the the gold labels of each task.
		/// Gold labels that are not consistent are returned as null
		/// Labels are returned as indexed by task index
		/// </summary>
		/// <returns>The gold labels per task index.</returns>
		public int?[] GetGoldLabelsPerTaskIndex()
		{
			return Data.GroupBy(d => TaskIdToIndex[d.TaskId]).
				OrderBy(g => g.Key).
				Select(t => t.GroupBy(d => d.GoldLabel).Where(d => d.Key != null)).
				Select(gold_d =>
					{
						int count = gold_d.Distinct().Count();
						if (count == 1)
						{
							int? gold = gold_d.First().First().GoldLabel;
							if (gold != null)
								gold = gold.Value - LabelMin;
							return gold;
						}
						else
							return (int?)null;
					}).ToArray();
		}
		/// <summary>
		/// Gets the random label per task identifier.
		/// </summary>
		/// <returns>The random label per task identifier.</returns>
		/// <param name="data">Data.</param>
        public Dictionary<string, int?> GetRandomLabelPerTaskId(IList<Datum> data)
        {
            // Labels are returned as indexed by task index
            return data.GroupBy(d => d.TaskId).
              Select(collection =>
              {
                  int r = Rand.Int(0, collection.Count() - 1);
                  return new Tuple<string, int?>(collection.Key, (int?)collection.ToArray()[r].WorkerLabel);
              }).ToDictionary(tup => tup.Item1, tup => tup.Item2);
        }
        /// <summary>
        /// For each task, gets the majority vote label if it is unique.
        /// </summary>
        /// <returns>The list of majority vote labels.</returns>
        public int?[] GetMajorityVotesPerTaskIndex()
        {
            return Data.GroupBy(d => TaskIdToIndex[d.TaskId]).
              OrderBy(g => g.Key).
              Select(t => t.GroupBy(d => d.WorkerLabel - LabelMin).
                  Select(g => new { label = g.Key, count = g.Count() })).
                  Select(arr =>
                  {
                      int max = arr.Max(a => a.count);
                      int[] majorityLabs = arr.Where(a => a.count == max).Select(a => a.label).ToArray();
                      if (majorityLabs.Length == 1)
                          return (int?)majorityLabs[0];
                      else
                      {
                          //return random label;
                          int r = Rand.Int(0, majorityLabs.Length);
                          return (int?)majorityLabs[0];
                      }
                  }).ToArray();
        }
        /// <summary>
        /// For each task Id, gets the majority vote label if it is unique.
        /// </summary>
        /// <returns>The dictionary of majority vote labels indexed by task id.</returns>
        public Dictionary<string, int?> GetMajorityVotesPerTaskId(IList<Datum> data)
        {
            Dictionary<string, int?> majorityVotesPerTaskId = new Dictionary<string, int?>();
			int?[] majorityVotes = GetMajorityVotesPerTaskIndex();
            foreach (Datum d in data)
            {
                if (!majorityVotesPerTaskId.ContainsKey(d.TaskId))
                    majorityVotesPerTaskId[d.TaskId] = majorityVotes[TaskIdToIndex[d.TaskId]];
            }
            return majorityVotesPerTaskId;
        }
    }
}