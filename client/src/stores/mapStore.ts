import { makeAutoObservable } from 'mobx';

class MapStore {
    mode: '3d' | 'flat' = '3d';
    activeLayer: 'confidence' | 'density' | 'eco_risk' = 'confidence';
    selectedBuilding: any = null;
    hoveredBuilding: any = null;
    targetPitch: number = 75;
    flyToTarget: any = null;
    heightRange: [number, number] = [0, 462];

    constructor() {
        makeAutoObservable(this);
    }

    setMode(mode: '3d' | 'flat') {
        this.mode = mode;
        this.targetPitch = mode === '3d' ? 75 : 0;
    }

    setActiveLayer(layer: 'confidence' | 'density' | 'eco_risk') {
        this.activeLayer = layer;
    }

    setSelectedBuilding(f: any) {
        this.selectedBuilding = f;
    }

    setHoveredBuilding(f: any) {
        this.hoveredBuilding = f;
    }

    setHeightRange(range: [number, number]) {
        this.heightRange = range;
    }

    flyTo(target: any) {
        this.flyToTarget = target;
    }
}

export const mapStore = new MapStore();
