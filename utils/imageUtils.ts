import JSZip from 'jszip';
import { ProcessedImage, AVAILABLE_CONFIGS } from '../types';

export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
};

export const downloadImage = (base64: string, filename: string) => {
  const link = document.createElement('a');
  link.href = base64;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const downloadAllAsZip = async (queue: ProcessedImage[]) => {
  const zip = new JSZip();
  const folder = zip.folder("autobanner-assets");
  
  if (!folder) return;

  const completedItems = queue.filter(item => item.status === 'completed' && item.resultUrl);

  completedItems.forEach((item) => {
    const config = AVAILABLE_CONFIGS.find(c => c.id === item.configId);
    const filename = `${config?.label.replace(/\s+/g, '_')}_${config?.width}x${config?.height}.png`;
    
    // Remove base64 header for JSZip
    const base64Data = item.resultUrl!.replace(/^data:image\/(png|jpeg|jpg);base64,/, "");
    folder.file(filename, base64Data, { base64: true });
  });

  const content = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(content);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = "autobanner_batch_export.zip";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};