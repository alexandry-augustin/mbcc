using System;
using System.IO;
using MicrosoftResearch.Infer.Distributions;
using System.Linq;
using System.Collections.Generic;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Utils;
using models.mbcc;
using models.ibcc;
using stats;
using System.Text;
using MicrosoftResearch.Infer;

namespace utils
{
	class PrettyPrint
	{
		public static string ErrorTypeStr(ErrorType errorType)
		{
			string name = "";
			switch(errorType)
			{
			case ErrorType.EucledianDistance:
				name = "Eucledian Distance";
				break;
			case ErrorType.KLDivergence:
				name="KL-Divergence";
				break;
			case ErrorType.MeanSquaredError:
				name="MeanSquaredError";
				break;
			default:
				name="N/A";
				break;
			}
			return name;
		}
		public static string SpammingStrategyStr(SpammingStrategy strategy)
		{
			string name = "";
			switch(strategy)
			{
			case SpammingStrategy.SingleLabel:
				name = "Single Label";
				break;
			case SpammingStrategy.Uniform:
				name="Uniform";
				break;
			case SpammingStrategy.UniformWithPrior:
				name="Uniform with prior";
				break;
			case SpammingStrategy.SparesUniformWithPrior:
				name="Uniform with prior with spares assignement";
				break;
			default:
				name="N/A";
				break;
			}
			return name;
		}
		public static string InferenceAlgorithmStr(IAlgorithm algo)
		{
			string name = "";
			if (algo.GetType () == typeof(ExpectationPropagation))
				name = "ExpectationPropagation";
			else if(algo.GetType () == typeof(VariationalMessagePassing))
				name = "VariationalMessagePassing";
			else if(algo.GetType () == typeof(GibbsSampling))
				name = "GibbsSampling";
			else
				name="N/A";

			return name;
		}
		/// <summary>
		/// Convert double to string for csv file
		/// </summary>
		/// <param name="value">value to convert</param>
		public static string ToCsv(double? value, string na="NA")
		{
			if(value==null)
				return na;

			if (Double.IsNaN((double)value))
				return na;

			return ((Double)value).ToString ("0.0000");
		}
	}
}