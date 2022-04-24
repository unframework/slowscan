import React from 'react';
import ReactDOM from 'react-dom';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';

import { MainStage } from './MainStage';

import './index.css';

const App: React.FC = () => {
  return (
    <Canvas
      style={{ height: '100vh' }}
      gl={{
        alpha: false,
        toneMapping: THREE.ACESFilmicToneMapping,
        toneMappingExposure: 0.9,

        outputEncoding: THREE.sRGBEncoding
      }}
    >
      <React.Suspense fallback={null}>
        <MainStage />
      </React.Suspense>
    </Canvas>
  );
};

ReactDOM.render(React.createElement(App), document.getElementById('root'));
