import React, { useState, useMemo } from 'react';
import { AVAILABLE_CONFIGS, AppState, ProcessedImage, Category } from './types';
import { processBannerLocally } from './services/localProcessingService';
import { downloadAllAsZip } from './utils/imageUtils';
import { Uploader } from './components/Uploader';
import { ResultCard } from './components/ResultCard';

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>({
    uploadedImage: null,
    selectedConfigs: [],
    queue: [],
    isProcessing: false,
  });

  // Group configs by category
  const configsByCategory = useMemo(() => {
    const groups: Record<string, typeof AVAILABLE_CONFIGS> = {};
    AVAILABLE_CONFIGS.forEach(conf => {
        if (!groups[conf.category]) groups[conf.category] = [];
        groups[conf.category].push(conf);
    });
    return groups;
  }, []);

  const handleImageSelected = (base64: string) => {
    setAppState(prev => ({ ...prev, uploadedImage: base64, queue: [] }));
  };

  const toggleConfig = (id: string) => {
    setAppState(prev => {
      const isSelected = prev.selectedConfigs.includes(id);
      return {
        ...prev,
        selectedConfigs: isSelected 
          ? prev.selectedConfigs.filter(c => c !== id)
          : [...prev.selectedConfigs, id]
      };
    });
  };

  const selectAllInCategory = (category: string) => {
      const idsInCategory = AVAILABLE_CONFIGS.filter(c => c.category === category).map(c => c.id);
      setAppState(prev => {
          const allSelected = idsInCategory.every(id => prev.selectedConfigs.includes(id));
          if (allSelected) {
              return { ...prev, selectedConfigs: prev.selectedConfigs.filter(id => !idsInCategory.includes(id)) };
          } else {
              const newSelection = new Set([...prev.selectedConfigs, ...idsInCategory]);
              return { ...prev, selectedConfigs: Array.from(newSelection) };
          }
      });
  };

  const selectAllGlobal = () => {
      setAppState(prev => {
          if (prev.selectedConfigs.length === AVAILABLE_CONFIGS.length) {
              return { ...prev, selectedConfigs: [] };
          }
          return { ...prev, selectedConfigs: AVAILABLE_CONFIGS.map(c => c.id) };
      });
  };

  const startProcessing = async () => {
    if (!appState.uploadedImage) return;

    // 1. Create initial queue state
    const newQueue: ProcessedImage[] = appState.selectedConfigs.map(configId => ({
      id: Math.random().toString(36).substring(7),
      originalImage: appState.uploadedImage!,
      configId,
      status: 'pending'
    }));

    setAppState(prev => ({ ...prev, queue: newQueue, isProcessing: true }));

    // 2. Define the processing function
    const processItem = async (item: ProcessedImage) => {
      const config = AVAILABLE_CONFIGS.find(c => c.id === item.configId);
      if (!config) return;

      updateItemStatus(item.id, 'processing');
      try {
        // Use the new Local Processing Service
        const resultBase64 = await processBannerLocally(
          item.originalImage,
          config.width,
          config.height,
          config.layoutRule
        );
        updateItemStatus(item.id, 'completed', resultBase64);
      } catch (error) {
        console.error(error);
        updateItemStatus(item.id, 'failed', undefined, "Processing failed");
      }
    };

    // 3. Process sequentially to manage CPU load better in browser
    // Parallel execution of Wasm background removal can crash the browser tab
    for (const item of newQueue) {
        await processItem(item);
    }

    setAppState(prev => ({ ...prev, isProcessing: false }));
  };

  const updateItemStatus = (itemId: string, status: ProcessedImage['status'], resultUrl?: string, error?: string) => {
    setAppState(prev => ({
      ...prev,
      queue: prev.queue.map(item => 
        item.id === itemId ? { ...item, status, resultUrl, error } : item
      )
    }));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-6 md:p-12">
      
      <header className="max-w-7xl mx-auto mb-12 text-center md:text-left border-b border-slate-800 pb-8">
        <div className="flex flex-col md:flex-row items-center justify-between">
            <div>
                <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent mb-2">
                AutoBanner <span className="text-white">Local</span>
                </h1>
                <p className="text-slate-400 max-w-2xl text-lg">
                100% On-Device Processing. Secure, Private, Free.
                </p>
            </div>
            <div className="mt-6 md:mt-0 flex flex-col items-end gap-2">
                 <div className="inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-slate-900 border border-slate-700">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                    <span className="text-xs font-mono text-slate-400">WASM / Canvas Engine Active</span>
                 </div>
            </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-12 gap-10">
        
        {/* Left Column: Input & Configuration */}
        <div className="xl:col-span-4 space-y-8 h-fit sticky top-6">
          
          <section className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center">
                <span className="bg-emerald-600 w-6 h-6 rounded-full flex items-center justify-center text-xs mr-2">1</span>
                Upload Master Creative
            </h2>
            <Uploader onImageSelected={handleImageSelected} />
            {appState.uploadedImage && (
              <div className="mt-4">
                 <img src={appState.uploadedImage} alt="Uploaded" className="w-full h-32 object-cover rounded-lg opacity-75 border border-slate-700" />
              </div>
            )}
          </section>

          <section className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-semibold text-white flex items-center">
                    <span className="bg-emerald-600 w-6 h-6 rounded-full flex items-center justify-center text-xs mr-2">2</span>
                    Select Formats
                </h2>
                <button 
                    onClick={selectAllGlobal}
                    className="text-xs text-emerald-400 hover:text-emerald-300 font-medium"
                >
                    {appState.selectedConfigs.length === AVAILABLE_CONFIGS.length ? 'Deselect All' : 'Select All (10+)'}
                </button>
            </div>
            
            <div className="space-y-6 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
              {(Object.keys(configsByCategory) as Category[]).map(category => (
                <div key={category} className="space-y-3">
                    <div className="flex items-center justify-between sticky top-0 bg-slate-900/95 py-2 z-10 backdrop-blur-sm">
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">{category}</h3>
                        <button 
                            onClick={() => selectAllInCategory(category)}
                            className="text-[10px] bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded text-slate-300"
                        >
                            Toggle Group
                        </button>
                    </div>
                    <div className="grid grid-cols-1 gap-2">
                        {configsByCategory[category].map(config => (
                            <label 
                            key={config.id} 
                            className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all group ${
                                appState.selectedConfigs.includes(config.id) 
                                ? 'bg-emerald-900/20 border-emerald-500/50' 
                                : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                            }`}
                            >
                            <div className="flex items-center space-x-3">
                                <input 
                                type="checkbox" 
                                checked={appState.selectedConfigs.includes(config.id)}
                                onChange={() => toggleConfig(config.id)}
                                className="w-4 h-4 rounded text-emerald-500 focus:ring-emerald-500 bg-slate-800 border-slate-600"
                                />
                                <div>
                                <p className="font-medium text-sm text-white group-hover:text-emerald-200">{config.label}</p>
                                <p className="text-[10px] text-slate-500">{config.width}x{config.height}</p>
                                </div>
                            </div>
                            <div className="text-[10px] font-mono text-slate-600 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                                {config.width > config.height ? 'LAND' : config.width === config.height ? 'SQ' : 'PORT'}
                            </div>
                            </label>
                        ))}
                    </div>
                </div>
              ))}
            </div>
          </section>

          <button
            onClick={startProcessing}
            disabled={!appState.uploadedImage || appState.isProcessing || appState.selectedConfigs.length === 0}
            className={`w-full py-4 px-6 rounded-xl font-bold text-lg shadow-xl transform transition-all border border-transparent ${
              !appState.uploadedImage || appState.isProcessing || appState.selectedConfigs.length === 0
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-emerald-600 text-white hover:bg-emerald-500 hover:shadow-emerald-500/25 active:scale-95'
            }`}
          >
            {appState.isProcessing ? (
              <span className="flex items-center justify-center space-x-2">
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Processing locally...</span>
              </span>
            ) : (
              `Generate ${appState.selectedConfigs.length > 0 ? appState.selectedConfigs.length : ''} Variations`
            )}
          </button>
        </div>

        {/* Right Column: Results Gallery */}
        <div className="xl:col-span-8">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-white flex items-center">
                    <span className="bg-emerald-600 w-6 h-6 rounded-full flex items-center justify-center text-xs mr-2">3</span>
                    Generated Assets
                    {appState.queue.length > 0 && (
                        <span className="ml-3 text-xs font-mono bg-slate-800 px-2 py-1 rounded text-slate-400">
                            {appState.queue.filter(i => i.status === 'completed').length}/{appState.queue.length} Ready
                        </span>
                    )}
                </h2>
                {appState.queue.some(i => i.status === 'completed') && (
                    <button 
                        onClick={() => downloadAllAsZip(appState.queue)}
                        className="flex items-center space-x-2 bg-slate-700 hover:bg-slate-600 text-white text-sm px-4 py-2 rounded-lg transition-colors"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        <span>Download ZIP</span>
                    </button>
                )}
            </div>
            
            {appState.queue.length === 0 ? (
                <div className="h-[600px] border-2 border-dashed border-slate-800 rounded-3xl flex flex-col items-center justify-center text-slate-600 bg-slate-900/20">
                    <div className="p-4 bg-slate-900 rounded-full mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <p className="font-medium text-lg text-slate-500">Workspace Empty</p>
                    <p className="text-sm">Upload a master creative to begin local automation.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-3 gap-6">
                    {appState.queue.map(item => (
                        <ResultCard key={item.id} item={item} />
                    ))}
                </div>
            )}
        </div>

      </main>
    </div>
  );
};

export default App;