import React from 'react';
import { extend, ReactThreeFiber } from '@react-three/fiber';
import { ScreenQuad, shaderMaterial } from '@react-three/drei';

import vert from './MainViewQuad.vert.glsl';
import frag from './MainViewQuad.frag.glsl';

const MainShaderMaterial = shaderMaterial(
  {
    mouse: [0, 0],
    resolution: [100, 100],
    blueNoise: null
  },
  vert,
  frag
);

export const MainViewQuad: React.FC = () => {
  return (
    <ScreenQuad>
      <mainShaderMaterial />
    </ScreenQuad>
  );
};

extend({ MainShaderMaterial });

declare global {
  namespace JSX {
    interface IntrinsicElements {
      mainShaderMaterial: ReactThreeFiber.Object3DNode<
        typeof MainShaderMaterial,
        {}
      >;
    }
  }
}
