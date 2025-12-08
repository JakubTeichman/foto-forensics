/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly REACT_APP_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
