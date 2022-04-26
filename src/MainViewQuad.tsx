import React from 'react';
import {
  extend,
  useThree,
  useLoader,
  ReactThreeFiber
} from '@react-three/fiber';
import { ScreenQuad } from '@react-three/drei';
import * as THREE from 'three';

import vert from './MainViewQuad.vert.glsl';
import frag from './MainViewQuad.frag.glsl';
import blueNoiseURL from './bluenoise.png'; // from ShaderToy

class MainShaderMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: {
        mouse: new THREE.Uniform([0, 0]),
        resolution: new THREE.Uniform([100, 100]),
        blueNoise: new THREE.Uniform(null)
      }
    });
  }
}

export const MainViewQuad: React.FC = () => {
  const { size } = useThree();

  // texture from https://opengameart.org/content/metalstone-textures by Spiney
  const bnTexture = useLoader(THREE.TextureLoader, blueNoiseURL);
  bnTexture.wrapS = THREE.RepeatWrapping;
  bnTexture.wrapT = THREE.RepeatWrapping;

  return (
    <ScreenQuad>
      <mainShaderMaterial
        uniforms-mouse-value={[0, size.height * 0.05]}
        uniforms-resolution-value={[size.width, size.height]}
        uniforms-blueNoise-value={bnTexture}
      />
    </ScreenQuad>
  );
};

extend({ MainShaderMaterial });

declare global {
  namespace JSX {
    interface IntrinsicElements {
      mainShaderMaterial: ReactThreeFiber.Object3DNode<
        MainShaderMaterial,
        typeof MainShaderMaterial
      >;
    }
  }
}
