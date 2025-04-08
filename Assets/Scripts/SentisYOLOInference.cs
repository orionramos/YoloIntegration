using UnityEngine;
using Unity.Sentis;
using System.Collections;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Linq;

public class SentisYOLOv8Inference : MonoBehaviour
{
    // Configuración del modelo
    [Header("Model Configuration")]
    public ModelAsset modelAsset;

    // Referencias de la interfaz: video en tiempo real y capa de overlay para inferencias, y texto para resultados.
    [Header("UI References")]
    public RawImage videoDisplay;      // Muestra el video en vivo.
    public RawImage overlayDisplay;    // Se superpone para dibujar solo los rectángulos de inferencia.
    public Text resultsText;           // Muestra el listado de detecciones.

    // Parámetros de inferencia
    [Header("Inference Settings")]
    [Range(0, 1)] public float confidenceThreshold = 0.5f; // Umbral de confianza para considerar una detección.
    [Range(0, 1)] public float iouThreshold = 0.5f;          // Umbral de IoU para la supresión de no máximos.

    // Configuración de los bounding boxes
    [Header("Bounding Box Settings")]
    public Color boundingBoxColor = Color.red;  // Color del rectángulo.
    public float boundingBoxLineWidth = 5f;       // Grosor del rectángulo.

    // Objeto a resaltar (se configura desde la interfaz de Unity)
    [Header("Highlight Object")]
    public string targetObject = "person"; // Cambia este valor en el Inspector para resaltar otro objeto

    // Dimensiones de entrada del modelo
    private const int MODEL_INPUT_WIDTH = 640;
    private const int MODEL_INPUT_HEIGHT = 640;

    // Etiquetas para la detección (COCO, por ejemplo) // modificar y entrenar personalizado
    private string[] labels = new string[] {
       "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
       "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
       "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
       "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
       "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
       "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
       "hair drier", "toothbrush"
    };

    // Variables internas para el modelo y la cámara.
    private Model runtimeModel;
    private Worker worker;
    private Tensor<float> inputTensor;
    private WebCamTexture webcamTexture;

    void Start()
    {
        // Se inicia la cámara del dispositivo.
        webcamTexture = new WebCamTexture();
        if (videoDisplay != null)
        {
            // Asigna el WebCamTexture al RawImage para mostrar el video.
            videoDisplay.texture = webcamTexture;
            videoDisplay.material.mainTexture = webcamTexture;
        }
        webcamTexture.Play();

        // Se inicia la corrutina que realiza la inferencia periódicamente.
        StartCoroutine(RunInferenceCoroutine());
    }

    // Redimensiona una Texture2D a las dimensiones requeridas por el modelo.
    Texture2D ResizeTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        // Se crea un RenderTexture con las dimensiones deseadas.
        RenderTexture rt = new RenderTexture(targetWidth, targetHeight, 24);
        RenderTexture.active = rt;
        Graphics.Blit(source, rt); // Se copia la imagen al RenderTexture.
        // Se crea una nueva Texture2D y se leen los píxeles del RenderTexture.
        Texture2D result = new Texture2D(targetWidth, targetHeight);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();
        RenderTexture.active = null;
        Destroy(rt); // Se destruye el RenderTexture para liberar recursos.
        return result;
    }

    // Corrutina que realiza la inferencia periódicamente.
    IEnumerator RunInferenceCoroutine()
    {
        while (true)
        {
            // Espera al final del frame para capturar la imagen actual de la cámara.
            yield return new WaitForEndOfFrame();

            // Captura la imagen actual del WebCamTexture y la convierte en Texture2D.
            Texture2D frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
            frameTexture.SetPixels(webcamTexture.GetPixels());
            frameTexture.Apply();

            // Se redimensiona el frame al tamaño que requiere el modelo.
            Texture2D processTexture = ResizeTexture(frameTexture, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
            Destroy(frameTexture); // Se libera la textura temporal.

            Tensor<float> outputTensor = null;

            // Se intenta cargar el modelo y preparar la inferencia.
            try
            {
                runtimeModel = ModelLoader.Load(modelAsset);
                // Convierte la textura en un tensor de entrada.
                inputTensor = TextureConverter.ToTensor(processTexture, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 3);
                worker = new Worker(runtimeModel, BackendType.GPUCompute);
                worker.Schedule(inputTensor); // Inicia la inferencia.
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Inference failed: {e.Message}");
                continue;
            }

            // Breve espera para que la inferencia se complete.
            yield return new WaitForSeconds(0.1f);

            // Se intenta obtener la salida del modelo.
            try
            {
                outputTensor = worker.PeekOutput() as Tensor<float>;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error retrieving inference output: {e.Message}");
            }

            if (outputTensor != null)
            {
                // Procesa la salida y obtiene un overlay con solo los rectángulos.
                Texture2D overlay = ProcessYOLOv8Output(outputTensor, processTexture);
                if (overlayDisplay != null)
                {
                    overlayDisplay.texture = overlay; // Actualiza el overlay en la interfaz.
                }
            }
            else
            {
                Debug.LogError("Failed to retrieve inference output");
            }

            CleanupResources(); // Libera recursos usados en la inferencia.
            // Espera 3 segundos (o el intervalo que hayas definido) antes de la siguiente inferencia.
            yield return new WaitForSeconds(3f);
        }
    }

    // Procesa la salida del modelo y dibuja los bounding boxes en un overlay transparente.
    Texture2D ProcessYOLOv8Output(Tensor<float> outputTensor, Texture2D processTexture)
    {
        // Descarga la salida del tensor en un arreglo de floats.
        float[] rawOutput = outputTensor.DownloadToArray();
        List<Detection> detections = new List<Detection>();

        // Se asume que la salida tiene forma [1,84,8400]; se recorre cada posible detección.
        for (int i = 0; i < 8400; i++)
        {
            float maxScore = 0;
            int classId = -1;
            // Se obtienen las puntuaciones para cada clase (comenzando en la posición 4).
            for (int c = 4; c < 84; c++)
            {
                float score = rawOutput[c * 8400 + i];
                if (score > maxScore)
                {
                    maxScore = score;
                    classId = c - 4;
                }
            }

            // Se obtienen las coordenadas (x, y, w, h).
            float x = rawOutput[0 * 8400 + i];
            float y = rawOutput[1 * 8400 + i];
            float w = rawOutput[2 * 8400 + i];
            float h = rawOutput[3 * 8400 + i];

            // Se filtra por umbral de confianza.
            if (maxScore > confidenceThreshold)
            {
                detections.Add(new Detection
                {
                    ClassId = classId,
                    Confidence = maxScore,
                    X = x,
                    Y = y,
                    Width = w,
                    Height = h
                });
            }
        }

        // Se aplica la supresión de no máximos (NMS) para eliminar detecciones redundantes.
        detections = ApplyNMS(detections);
        // Se dibujan los bounding boxes sobre un overlay transparente.
        return DrawDetections(processTexture, detections);
    }

    // Dibuja los bounding boxes sobre una textura transparente.
    Texture2D DrawDetections(Texture2D processTexture, List<Detection> detections)
    {
        string resultsMessage = "Detected Objects:\n";

        // Crea una textura transparente del mismo tamaño que la imagen de entrada.
        Texture2D overlayTexture = new Texture2D(processTexture.width, processTexture.height, TextureFormat.RGBA32, false);
        Color[] transparentPixels = new Color[processTexture.width * processTexture.height];
        for (int i = 0; i < transparentPixels.Length; i++)
            transparentPixels[i] = new Color(0, 0, 0, 0);
        overlayTexture.SetPixels(transparentPixels);
        overlayTexture.Apply();

        // Procesa hasta 10 detecciones.
        foreach (var detection in detections.Take(10))
        {
            string label = (detection.ClassId >= 0 && detection.ClassId < labels.Length)
                ? labels[detection.ClassId]
                : $"Unknown ({detection.ClassId})";
            resultsMessage += $"{label}: {detection.Confidence:F2} - Pos: ({detection.X:F2}, {detection.Y:F2}), Size: {detection.Width:F2}x{detection.Height:F2}\n";

            // Solo dibuja el bounding box si la detección coincide con el objeto a resaltar.
            if (label.Equals(targetObject))
            {
                DrawBoundingBox(overlayTexture, detection, boundingBoxColor);
            }
        }

        // Actualiza el texto de resultados.
        if (resultsText != null)
            resultsText.text = resultsMessage;
        Debug.Log(resultsMessage);

        return overlayTexture;
    }

    // Aplica Non-Maximum Suppression (NMS) para filtrar detecciones redundantes.
    List<Detection> ApplyNMS(List<Detection> detections)
    {
        List<Detection> filteredDetections = new List<Detection>();
        while (detections.Count > 0)
        {
            // Ordena las detecciones por confianza.
            detections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
            var best = detections[0];
            filteredDetections.Add(best);
            detections.RemoveAt(0);
            // Elimina detecciones que se solapen en exceso con la mejor.
            detections.RemoveAll(d => CalculateIOU(best, d) > iouThreshold);
        }
        return filteredDetections;
    }

    // Calcula la Intersección sobre Unión (IoU) entre dos detecciones.
    float CalculateIOU(Detection a, Detection b)
    {
        float xA = Mathf.Max(a.X - a.Width / 2, b.X - b.Width / 2);
        float yA = Mathf.Max(a.Y - a.Height / 2, b.Y - b.Height / 2);
        float xB = Mathf.Min(a.X + a.Width / 2, b.X + b.Width / 2);
        float yB = Mathf.Min(a.Y + a.Height / 2, b.Y + b.Height / 2);

        float intersectionArea = Mathf.Max(0, xB - xA) * Mathf.Max(0, yB - yA);
        float unionArea = a.Width * a.Height + b.Width * b.Height - intersectionArea;
        return intersectionArea / unionArea;
    }

    // Dibuja un bounding box (rectángulo) en la textura.
    void DrawBoundingBox(Texture2D texture, Detection detection, Color boxColor)
    {
        int xMin = Mathf.Clamp((int)(detection.X - detection.Width / 2), 0, texture.width - 1);
        int yMin = Mathf.Clamp((int)(detection.Y - detection.Height / 2), 0, texture.height - 1);
        int xMax = Mathf.Clamp((int)(detection.X + detection.Width / 2), 0, texture.width - 1);
        int yMax = Mathf.Clamp((int)(detection.Y + detection.Height / 2), 0, texture.height - 1);

        // Dibuja las cuatro líneas del rectángulo.
        DrawLine(texture, xMin, yMin, xMax, yMin, boxColor, (int)boundingBoxLineWidth);
        DrawLine(texture, xMin, yMax, xMax, yMax, boxColor, (int)boundingBoxLineWidth);
        DrawLine(texture, xMin, yMin, xMin, yMax, boxColor, (int)boundingBoxLineWidth);
        DrawLine(texture, xMax, yMin, xMax, yMax, boxColor, (int)boundingBoxLineWidth);
        texture.Apply();
    }

    // Dibuja una línea en la textura utilizando el algoritmo de Bresenham.
    void DrawLine(Texture2D tex, int x0, int y0, int x1, int y1, Color color, int lineWidth)
    {
        int dx = Mathf.Abs(x1 - x0);
        int dy = Mathf.Abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            // Dibuja un cuadrado centrado en (x0, y0) para simular el grosor de la línea.
            for (int i = -lineWidth / 2; i < lineWidth / 2; i++)
            {
                for (int j = -lineWidth / 2; j < lineWidth / 2; j++)
                {
                    if (Vector2Int.Distance(Vector2Int.zero, new Vector2Int(i, j)) < lineWidth / 2)
                    {
                        int x = Mathf.Clamp(x0 + i, 0, tex.width - 1);
                        int y = Mathf.Clamp(y0 + j, 0, tex.height - 1);
                        tex.SetPixel(x, y, color);
                    }
                }
            }
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    // Libera los recursos utilizados por el tensor y el worker.
    void CleanupResources()
    {
        inputTensor?.Dispose();
        worker?.Dispose();
    }

    void OnDisable()
    {
        CleanupResources();
        if (webcamTexture != null && webcamTexture.isPlaying)
            webcamTexture.Stop();
    }

    // Clase para almacenar la información de cada detección.
    class Detection
    {
        public int ClassId { get; set; }
        public float Confidence { get; set; }
        public float X { get; set; }  // Coordenada X del centro.
        public float Y { get; set; }  // Coordenada Y del centro.
        public float Width { get; set; }
        public float Height { get; set; }
    }
}
