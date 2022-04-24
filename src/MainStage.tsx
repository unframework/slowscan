import React, { useState, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { OrthographicCamera, ScreenQuad } from '@react-three/drei';
import * as THREE from 'three';

import './MainShaderMaterial';

export const MainStage: React.FC = () => {
  const cameraRef = useRef<THREE.Object3D>();

  return (
    <group>
      <OrthographicCamera
        near={-1}
        far={1}
        left={-1}
        right={1}
        top={1}
        bottom={-1}
        makeDefault
        ref={cameraRef}
      />

      <ScreenQuad>
        <mainShaderMaterial />
      </ScreenQuad>
    </group>
  );
};
