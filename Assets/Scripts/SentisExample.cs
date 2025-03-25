using UnityEngine;
using Unity.Sentis;

public class SentisExample : MonoBehaviour
{
    public ModelAsset modelAsset;
    public Texture2D inputTexture;
    public float[] results;

    Model runtimeModel;
    Worker worker;


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        // Create input data as a tensor
        using Tensor inputTensor = TextureConverter.ToTensor(inputTexture, width: 640  , height: 640, channels: 3);

        // Create an engine
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        // Run the model with the input data
        worker.Schedule(inputTensor);
        // Get the result
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
        // outputTensor is still pending
        // Either read back the results asynchronously or do a blocking download call
        results = outputTensor.DownloadToArray();
    }
    void OnDisable()
    {
        // Tell the GPU we're finished with the memory the engine used
        worker.Dispose();
    }



}
