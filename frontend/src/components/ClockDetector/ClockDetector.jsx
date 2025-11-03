  import React, { useState } from "react";

  export const ClockDetector = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [time, setTime] = useState(null);
    const [steps, setSteps] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [handMode, setHandMode] = useState("auto"); 

    const handModeOptions = [
      { value: "auto", label: "Auto" },
      { value: "dark", label: "Dark hands" },
      { value: "light", label: "Light hands" },
    ];

    const handleUpload = (e) => {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setTime(null);
      setSteps(null);
      setError(null);
      setHandMode("auto");

      if (selectedFile) {
        const reader = new FileReader();
        reader.onloadend = () => setPreview(reader.result);
        reader.readAsDataURL(selectedFile);
      } else {
        setPreview(null);
      }
    };

    const handleDetect = async () => {
      if (!file) return;
      setLoading(true);
      setError(null);
      setTime(null);
      setSteps(null); 

      const formData = new FormData();
      formData.append("file", file);

      let apiUrl = "http://localhost:8000/detect";
      if (handMode !== "auto") {
          apiUrl += `?hand_mode=${handMode}`; 
      }

      try {
        const response = await fetch(apiUrl, {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
          const errorMessage = data.error || data.steps?.[0] || "Failed to process image due to an unknown error.";
          setError(errorMessage);
          if (data.visual_steps) {
              setSteps(data.visual_steps);
          }
          return;
        }

        if (data.hour === null || data.minute === null) {
          throw new Error(data.steps[0] || "Could not detect clock on the image");
        }

        setTime(data);
        setSteps(data.visual_steps || null);
      } catch (err) {
        console.error(err);
        setError(err.message);
      }

      setLoading(false);
    };

    return (
      <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
        <h1 className="text-3xl font-bold mb-6">Clock Time Detection</h1>

        <input
          type="file"
          onChange={handleUpload}
          accept="image/*"
          className="mb-4"
        />

        <div className="mb-4 w-full max-w-xs">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Hands Mode:
          </label>
          <select
            value={handMode}
            onChange={(e) => setHandMode(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md shadow-sm"
          >
            {handModeOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {preview && (
          <div className="mb-4 border rounded shadow-lg p-2 bg-white">
            <img
              src={preview}
              alt="Uploaded preview"
              className="max-w-sm max-h-80 object-contain"
            />
          </div>
        )}

        <button
          onClick={handleDetect}
          className="mb-4 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded shadow"
        >
          {loading ? "Processing..." : "Detect Time"}
        </button>

        {error && <div className="text-red-600 font-semibold mb-4">{error}</div>}

        {time && (
          <div className="mt-4 bg-white p-4 rounded shadow flex flex-col gap-2">
            <h2 className="text-xl font-semibold">Result:</h2>
            <div className="flex gap-4">
              <p className="text-lg">Часы: <span className="font-bold">{time.hour}</span></p>
              <p className="text-lg">Минуты: <span className="font-bold">{time.minute}</span></p>
            </div>
            <p className="text-sm text-gray-600 mt-1">
              Использованный режим: <span className="font-semibold uppercase">{time.mode || 'N/A'}</span>
            </p>
          </div>
        )}

        {/* Steps */}
        {steps && (
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {steps.face_detection && (
              <div className="border rounded shadow-lg p-2 bg-white">
                <h4 className="font-semibold mb-2">Detected Ellipse</h4>
                <img
                  src={`data:image/jpeg;base64,${steps.face_detection}`}
                  alt="Ellipse"
                  className="max-w-sm max-h-80 object-contain"
                />
              </div>
            )}
            {steps.hands_detection && (
              <div className="border rounded shadow-lg p-2 bg-white">
                <h4 className="font-semibold mb-2">Detected Hands</h4>
                <img
                  src={`data:image/jpeg;base64,${steps.hands_detection}`}
                  alt="Edges"
                  className="max-w-sm max-h-80 object-contain"
                />
              </div>
            )}
            {steps.final_result && (
              <div className="border rounded shadow-lg p-2 bg-white">
                <h4 className="font-semibold mb-2">Final</h4>
                <img
                  src={`data:image/jpeg;base64,${steps.final_result}`}
                  alt="Lines"
                  className="max-w-sm max-h-80 object-contain"
                />
              </div>
            )}
          </div>
        )}
      </div>
    );
  };
