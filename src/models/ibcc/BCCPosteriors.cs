using MicrosoftResearch.Infer.Distributions;
using System;

namespace models.ibcc
{
    /// <summary>
    /// The BCC posteriors class.
    /// </summary>
    [Serializable]
    public class IBCCPosteriors
    {
        /// <summary>
        /// The probabilities that generate the true labels of all the tasks.
        /// </summary>
        public Dirichlet BackgroundLabelProb;
        /// <summary>
        /// The probabilities of the true label of each task.
        /// </summary>
        public Discrete[] TrueLabel;
        /// <summary>
        /// The Dirichlet parameters of the confusion matrix of each worker.
        /// </summary>
        public Dirichlet[][] WorkerConfusionMatrix;
    }
}
