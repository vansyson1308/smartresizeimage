export enum AspectRatio {
  SQUARE = '1:1',
  PORTRAIT = '9:16',
  LANDSCAPE = '16:9',
  WIDE = '2:1',
  TALL = '1:4',
}

export type Category = 'Social Media' | 'Google Display Ads' | 'Video & Display';

export type LayoutRule = 'center' | 'bottom-stack' | 'left-anchor' | 'right-anchor';

export interface ResizeConfig {
  id: string;
  label: string;
  width: number;
  height: number;
  category: Category;
  description: string;
  layoutRule: LayoutRule;
}

export interface ProcessedImage {
  id: string;
  originalImage: string; // base64
  configId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  resultUrl?: string; // base64 or URL
  error?: string;
}

export interface AppState {
  uploadedImage: string | null; // base64
  selectedConfigs: string[];
  queue: ProcessedImage[];
  isProcessing: boolean;
}

export const AVAILABLE_CONFIGS: ResizeConfig[] = [
  // --- Social Media ---
  {
    id: 'social-square',
    label: 'Square Post',
    width: 1080,
    height: 1080,
    category: 'Social Media',
    description: 'Instagram/FB Feed',
    layoutRule: 'center'
  },
  {
    id: 'social-portrait',
    label: 'Portrait Post',
    width: 1080,
    height: 1350,
    category: 'Social Media',
    description: 'Instagram Grid',
    layoutRule: 'center'
  },
  {
    id: 'social-stories',
    label: 'Stories/Reels',
    width: 1080,
    height: 1920,
    category: 'Social Media',
    description: 'IG Stories, TikTok',
    layoutRule: 'bottom-stack'
  },
  {
    id: 'social-fb-cover',
    label: 'Facebook Cover',
    width: 820,
    height: 312,
    category: 'Social Media',
    description: 'Page Cover',
    layoutRule: 'left-anchor'
  },

  // --- Google Display Ads (GDN) ---
  {
    id: 'gdn-medium-rect',
    label: 'Medium Rectangle',
    width: 300,
    height: 250,
    category: 'Google Display Ads',
    description: '300x250',
    layoutRule: 'center'
  },
  {
    id: 'gdn-leaderboard',
    label: 'Leaderboard',
    width: 728,
    height: 90,
    category: 'Google Display Ads',
    description: '728x90 Header',
    layoutRule: 'left-anchor'
  },
  {
    id: 'gdn-wide-sky',
    label: 'Wide Skyscraper',
    width: 160,
    height: 600,
    category: 'Google Display Ads',
    description: '160x600 Sidebar',
    layoutRule: 'bottom-stack'
  },
  {
    id: 'gdn-large-rect',
    label: 'Large Rectangle',
    width: 336,
    height: 280,
    category: 'Google Display Ads',
    description: '336x280',
    layoutRule: 'center'
  },
  {
    id: 'gdn-billboard',
    label: 'Billboard',
    width: 970,
    height: 250,
    category: 'Google Display Ads',
    description: '970x250 Top',
    layoutRule: 'left-anchor'
  },

  // --- Video & Display ---
  {
    id: 'vid-full-hd',
    label: 'Full HD',
    width: 1920,
    height: 1080,
    category: 'Video & Display',
    description: '1920x1080',
    layoutRule: 'center'
  },
  {
    id: 'vid-landscape-ad',
    label: 'Landscape Ad',
    width: 1200,
    height: 628,
    category: 'Video & Display',
    description: 'LinkedIn/Native',
    layoutRule: 'right-anchor'
  }
];