import React, { useRef } from 'react';
import { fileToBase64 } from '../utils/imageUtils';

interface UploaderProps {
  onImageSelected: (base64: string) => void;
}

export const Uploader: React.FC<UploaderProps> = ({ onImageSelected }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      try {
        const base64 = await fileToBase64(file);
        onImageSelected(base64);
      } catch (error) {
        console.error("Error reading file", error);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
       const file = e.dataTransfer.files[0];
       try {
        const base64 = await fileToBase64(file);
        onImageSelected(base64);
      } catch (error) {
        console.error("Error reading file", error);
      }
    }
  };

  return (
    <div 
      className="border-2 border-dashed border-slate-600 rounded-xl p-12 text-center hover:border-indigo-500 hover:bg-slate-800/50 transition-all cursor-pointer group"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input 
        type="file" 
        accept="image/*" 
        className="hidden" 
        ref={fileInputRef}
        onChange={handleFileChange}
      />
      <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">Upload Banner Image</h3>
      <p className="text-slate-400 text-sm">Drag & drop or click to browse</p>
      <p className="text-slate-500 text-xs mt-2">Supports JPG, PNG, WEBP</p>
    </div>
  );
};