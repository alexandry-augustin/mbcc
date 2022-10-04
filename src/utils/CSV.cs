using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Collections;
using MicrosoftResearch.Infer.Distributions;
using models.mbcc;
using utils.mbcc;
using utils;

namespace utils
{
	public class csv
	{
		/// <summary>
		/// Loads the csv data file in the format (worker id, task id, worker label, gold).
		/// Used for images segmentation ratio datasets (iapr-tc12)
		/// </summary>
		/// <returns>List of parsed data.</returns>
		public static IList<Datum> LoadRawData2(
			string filename, 
			char sep=',', 
			bool skipHeader=false, 
			int maxLength=Int16.MaxValue)
		{
			var result = new List<Datum>();
			using (var reader = new StreamReader(filename))
			{
				string line;
				while ((line = reader.ReadLine()) != null && result.Count<maxLength)
				{
					if (skipHeader)
					{
						skipHeader = false;
						continue;
					}

					string[] str = line.Split(sep);
					if (str.Length != 4)
						continue;

					double workerProb = double.Parse(str[2], CultureInfo.InvariantCulture);
					Datum datum = new Datum()
					{
						WorkerId = str[0],
						TaskId = str[1],
						WorkerProb = workerProb
					};
						
					datum.GoldProb = double.Parse(str[3], CultureInfo.InvariantCulture);

					result.Add(datum);
				}
			}
			return result;
		}
		/// <summary>
		/// Loads the data file in the format (worker id, task id, worker label, ?gold label, ?TimeSpent).
		/// </summary>
		/// <returns>List of parsed data.</returns>
		public static IList<Datum> LoadData(
			string filename, 
			char sep=',', 
			bool skipHeader=false, 
			int maxLength=Int16.MaxValue)
		{
			var result = new List<Datum>();
			using (var reader = new StreamReader(filename))
			{
				string line;
				while ((line = reader.ReadLine()) != null && result.Count<maxLength)
				{
					if (skipHeader)
					{
						skipHeader = false;
						continue;
					}

					string[] str = line.Split(sep);
					int length = str.Length;
					if (length < 3 || length > 4)
					{
						continue;
					}

					int workerLabel = int.Parse(str[2]);
					Datum datum = new Datum()
					{
						WorkerId = str[0],
						TaskId = str[1],
						WorkerLabel = workerLabel
					};

					if (length >= 4 && !str[3].Equals("NaN"))
						datum.GoldLabel = int.Parse(str[3]);
					else
						datum.GoldLabel = null;

					result.Add(datum);
				}
			}
			return result;
		}
		/// <summary>
		/// Loads the data file in the format (!amt_annotation_ids, worker id, task id, worker label, gold label)
		/// Used fot SemEval2007 dataset.
		/// </summary>
		/// <returns>List of parsed data.</returns>
		private static IList<Datum> LoadRawData(
			string filename, 
			char sep=',', 
			bool skipHeader=false, 
			int maxLength=Int16.MaxValue)
        {
			List<Datum> result = new List<Datum>();
			using (var reader = new StreamReader(filename))
            {
				string line;
				while ((line = reader.ReadLine()) != null && result.Count<maxLength)
                {
					if (skipHeader)
					{
						skipHeader = false;
						continue;
					}

					var str = line.Split(sep);
                    int length = str.Length;
                    if (length != 5)
                    {
                        continue;
                    }
                    var datum = new Datum()
                    {
                        WorkerId = str[1],
                        TaskId = str[2],
                        WorkerLabel = int.Parse(str[3])
                    };

                    if (length >= 5 && !str[3].Equals("NaN"))
                        datum.GoldLabel = int.Parse(str[4]);
                    else
                        datum.GoldLabel = null;
						
					result.Add(datum);
            	}
            }
			return result;
        }
		/// <summary>
		/// Load the experimental data into a list of DatumMultiClass objects
		/// </summary>
		/// <returns>List of parsed data.</returns>
		public static List<DatumDist> LoadSemEvalData(
			string path, 
			bool skipUniform=false,
			bool skipPointMass=false)
		{
			//---------------------------------------------------------------
			// load all data
			//---------------------------------------------------------------
			IList<Datum> data = new List<Datum>();
			foreach(string prefix in new string[] { "anger", "disgust", "fear", "joy", "sadness", "surprise" })
			{
				string filename = string.Format("{0}{1}.standardized.tsv", path, prefix);
				data.AddRange(LoadRawData(filename, '\t', true, Int16.MaxValue));
			}
			//---------------------------------------------------------------
			// group data by taskId
			//---------------------------------------------------------------
			List<DatumDist> multiClassData = new List<DatumDist>();
			var workerGroup = data.GroupBy(d => d.WorkerId);
			int nbUniformSkipped = 0;
			int nbPointMassSkipped = 0;
			foreach(var wg in workerGroup)
			{
				string workerId = wg.Key;
				var workerTaskGroup = wg.GroupBy(g => g.TaskId);
				foreach (var wtg in workerTaskGroup) // Loops over the tasks judged by the worker
				{
					var datum = new DatumDist ();
					datum.TaskId = wtg.Key;
					datum.WorkerId = workerId;
					int[] arr = wtg.Select (d => d.WorkerLabel).ToArray ();
					datum.WorkerArray = arr;
					//normalise judgment
					if (arr.Sum () > 0)
					{
						if (skipPointMass && utils.Utils.isPointMass (datum.WorkerArray))
						{
							nbPointMassSkipped += 1;
							continue;
						}
						else
						{
							datum.WorkerDistr = Vector.FromArray (arr.Select (val => (double)val / arr.Sum ()).ToArray ());
						}

					}
					else // array of zeros
					{	
						if (skipUniform)
						{
							nbUniformSkipped += 1;
							continue;
						}
						else
						{
							//turn it into a uniform distribution
							datum.WorkerDistr = Vector.FromArray (Util.ArrayInit<double> (arr.Length, val => 1.0 / arr.Length)); 
						}
					}

					int?[] goldArr = wtg.Select(d => d.GoldLabel).ToArray();
					datum.GoldArr = goldArr;

					var normGoldArr = goldArr.Select(val => (double) val / goldArr.Sum()).ToArray();
					if (goldArr.Sum() > 0)
						datum.GoldDistr = Vector.FromArray(normGoldArr.Select(val => val == null ? 0 : (double) val).ToArray());

					multiClassData.Add(datum);
				}
			}

			Console.WriteLine ("{0} uniform judgments skipped.", nbUniformSkipped);
			Console.WriteLine ("{0} point-mass judgments skipped.", nbPointMassSkipped);

			return multiClassData;
		}
		/// <summary>
		/// Load the experimental data into a list of DatumMultiClass objects
		/// </summary>
		/// <returns>List of parsed data.</returns>
		public static List<DatumDist> LoadIAPRTC12Data(string path)
		{
			//---------------------------------------------------------------
			// load all data
			//---------------------------------------------------------------
			IList<Datum> data = new List<Datum>();
			foreach(string prefix in new string[] { "human", "animal", "food", "landscape-nature", "man-made", "other" })
			{
				string filename = string.Format("{0}{1}.csv", path, prefix);
				data.AddRange(LoadRawData2(filename, ',', true, Int16.MaxValue));
			}
			//---------------------------------------------------------------
			// group data by taskId
			//---------------------------------------------------------------
			List<DatumDist> multiClassData = new List<DatumDist>();
			var workerGroup = data.GroupBy(d => d.WorkerId);
			foreach(var wg in workerGroup)
			{
				string workerId = wg.Key;
				var workerTaskGroup = wg.GroupBy(g => g.TaskId);
				foreach (var wtg in workerTaskGroup) // Loops over the tasks judged by the worker
				{
					var datum = new DatumDist ();
					datum.TaskId = wtg.Key;
					datum.WorkerId = workerId;
					datum.WorkerDistr = Vector.FromArray(wtg.Select (d => d.WorkerProb).ToArray ());
					datum.GoldDistr = Vector.FromArray(wtg.Select (d => d.GoldProb).ToArray());

					multiClassData.Add(datum);
				}
			}

			return multiClassData;
		}
		/// <summary>
		/// Load the experimental data into a list of DatumMultiClass objects
		/// </summary>
		/// <returns>List of parsed data.</returns>
		public static List<DatumDist> LoadFlagsData(string path)
		{
			//---------------------------------------------------------------
			// load all data
			//---------------------------------------------------------------
			IList<Datum> data = new List<Datum>();
			foreach(string prefix in new string[] { "white", "red", "blue", "black", "yellow", "green", "orange", "purple", "gray", "brown" })
			{
				string filename = string.Format("{0}{1}.csv", path, prefix);
				data.AddRange(LoadRawData2(filename, ',', true, Int16.MaxValue));
			}
			//---------------------------------------------------------------
			// group data by taskId
			//---------------------------------------------------------------
			List<DatumDist> multiClassData = new List<DatumDist>();
			var workerGroup = data.GroupBy(d => d.WorkerId);
			foreach(var wg in workerGroup)
			{
				string workerId = wg.Key;
				var workerTaskGroup = wg.GroupBy(g => g.TaskId);
				foreach (var wtg in workerTaskGroup) // Loops over the tasks judged by the worker
				{
					var datum = new DatumDist ();
					datum.TaskId = wtg.Key;
					datum.WorkerId = workerId;
					datum.WorkerDistr = Vector.FromArray(wtg.Select (d => d.WorkerProb).ToArray ());
					datum.GoldDistr = Vector.FromArray(wtg.Select (d => d.GoldProb).ToArray());

					multiClassData.Add(datum);
				}
			}

			return multiClassData;
		}
	}
}

