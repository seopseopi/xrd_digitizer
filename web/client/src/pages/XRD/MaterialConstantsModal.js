import React, { useState, useEffect } from 'react';
import { 
  getDefaultElasticConstants, 
  getDislocationContrastPreset,
  estimateLatticeConstantFromPeaks,
  estimateStructureFromPeaks
} from '../../analysis/core/xrd/dislocationAnalysis';

/**
 * 재료 물성 상수 입력 모달
 * @param {boolean} isOpen - 모달 열림 여부
 * @param {Function} onClose - 모달 닫기 함수
 * @param {Function} onSave - 저장 함수 (materialConstants 객체 전달)
 * @param {Object} initialValues - 초기값
 * @param {Array} indexedPeaks - 인덱싱된 피크 데이터 (격자 상수 추정용)
 * @param {number} wavelength - X선 파장 (Å, 기본값: 1.5405)
 */
const MaterialConstantsModal = ({ 
  isOpen, 
  onClose, 
  onSave, 
  initialValues = null,
  indexedPeaks = [],
  wavelength = 1.5405 // Å
}) => {
  const [formData, setFormData] = useState({
    structure: 'fcc',
    latticeConstant: '',
    elasticConstants: {
      C11: '204.6',
      C12: '137.7',
      C44: '126.2'
    },
    contrastFactors: {
      qEdge: '1.71',
      qScrew: '2.46',
      Ch00Edge: '0.256',
      Ch00Screw: '0.305',
    },
    wavelength: String(wavelength / 10), // Å를 nm로 변환
    composition: '', // 조성 정보
    sampleInfo: '' // 추가 시편 정보
  });

  const [useDefaults, setUseDefaults] = useState(true);
  const [estimatedLatticeConstant, setEstimatedLatticeConstant] = useState(null);
  const [estimatedStructure, setEstimatedStructure] = useState(null);

  // XRD 피크로부터 격자 상수 및 결정 구조 추정
  useEffect(() => {
    if (isOpen && indexedPeaks && indexedPeaks.length > 0) {
      const wavelengthNm = parseFloat(formData.wavelength) || wavelength / 10;
      
      // 격자 상수 추정
      const estimatedA = estimateLatticeConstantFromPeaks(indexedPeaks, wavelengthNm);
      if (estimatedA) {
        setEstimatedLatticeConstant(estimatedA);
        // 추정값이 있으면 자동으로 입력
        if (!formData.latticeConstant || formData.latticeConstant === '') {
          setFormData(prev => ({ ...prev, latticeConstant: String(estimatedA.toFixed(3)) }));
        }
      }
      
      // 결정 구조 추정
      const estimatedStruct = estimateStructureFromPeaks(indexedPeaks);
      if (estimatedStruct) {
        setEstimatedStructure(estimatedStruct);
        // 추정값이 있으면 자동으로 선택
        if (!initialValues || !initialValues.structure) {
          setFormData(prev => ({ ...prev, structure: estimatedStruct }));
          handleStructureChange(estimatedStruct);
        }
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, indexedPeaks]);

  // 초기값 설정
  useEffect(() => {
    if (initialValues) {
      setFormData({
        structure: initialValues.structure || 'fcc',
        latticeConstant: String(initialValues.latticeConstant || ''),
        elasticConstants: {
          C11: String(initialValues.elasticConstants?.C11 || '204.6'),
          C12: String(initialValues.elasticConstants?.C12 || '137.7'),
          C44: String(initialValues.elasticConstants?.C44 || '126.2')
        },
        contrastFactors: {
          qEdge: String(initialValues.contrastFactors?.qEdge || '1.71'),
          qScrew: String(initialValues.contrastFactors?.qScrew || '2.46'),
          Ch00Edge: String(initialValues.contrastFactors?.Ch00Edge || '0.256'),
          Ch00Screw: String(initialValues.contrastFactors?.Ch00Screw || '0.305'),
        },
        wavelength: String(initialValues.wavelength || wavelength / 10),
        composition: initialValues.composition || '',
        sampleInfo: initialValues.sampleInfo || ''
      });
      setUseDefaults(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialValues]);

  // 결정 구조 변경 시 기본값 업데이트
  const handleStructureChange = (structure) => {
    const defaults = getDefaultElasticConstants(structure);
    const contrastDefaults = getDislocationContrastPreset(structure);
    const newFormData = {
      ...formData,
      structure,
      elasticConstants: {
        C11: String(defaults.C11),
        C12: String(defaults.C12),
        C44: String(defaults.C44)
      },
      contrastFactors: {
        qEdge: String(contrastDefaults.qEdge),
        qScrew: String(contrastDefaults.qScrew),
        Ch00Edge: String(contrastDefaults.Ch00Edge),
        Ch00Screw: String(contrastDefaults.Ch00Screw)
      }
    };

    // 격자 상수 기본값도 변경
    if (structure === 'fcc') {
      newFormData.latticeConstant = '3.615'; // SUS304
    } else if (structure === 'bcc') {
      newFormData.latticeConstant = '2.866'; // α-Fe
    }

    setFormData(newFormData);
  };

  // 기본값 사용 토글
  const handleUseDefaultsToggle = (checked) => {
    setUseDefaults(checked);
    if (checked) {
      const defaults = getDefaultElasticConstants(formData.structure);
      setFormData({
        ...formData,
        elasticConstants: {
          C11: String(defaults.C11),
          C12: String(defaults.C12),
          C44: String(defaults.C44)
        }
      });
    }
  };

  // 피크 정보로부터 격자 상수 자동 계산
  const handleEstimateFromPeaks = () => {
    if (!indexedPeaks || indexedPeaks.length < 2) {
      alert('격자 상수 추정을 위해서는 최소 2개의 인덱싱된 피크가 필요합니다.');
      return;
    }

    const wavelengthNm = parseFloat(formData.wavelength) || wavelength / 10;
    const estimatedA = estimateLatticeConstantFromPeaks(indexedPeaks, wavelengthNm);
    
    if (estimatedA) {
      setEstimatedLatticeConstant(estimatedA);
      setFormData(prev => ({ ...prev, latticeConstant: String(estimatedA.toFixed(3)) }));
      alert(`격자 상수가 추정되었습니다: ${estimatedA.toFixed(3)} Å`);
    } else {
      alert('격자 상수 추정에 실패했습니다. 피크 정보를 확인해주세요.');
    }

    // 결정 구조도 추정
    const estimatedStruct = estimateStructureFromPeaks(indexedPeaks);
    if (estimatedStruct) {
      setEstimatedStructure(estimatedStruct);
      setFormData(prev => ({ ...prev, structure: estimatedStruct }));
      handleStructureChange(estimatedStruct);
    }
  };

  // 저장 핸들러
  const handleSave = () => {
    const materialConstants = {
      structure: formData.structure,
      latticeConstant: parseFloat(formData.latticeConstant),
      elasticConstants: {
        C11: parseFloat(formData.elasticConstants.C11),
        C12: parseFloat(formData.elasticConstants.C12),
        C44: parseFloat(formData.elasticConstants.C44)
      },
      contrastFactors: {
        qEdge: parseFloat(formData.contrastFactors.qEdge),
        qScrew: parseFloat(formData.contrastFactors.qScrew),
        Ch00Edge: parseFloat(formData.contrastFactors.Ch00Edge),
        Ch00Screw: parseFloat(formData.contrastFactors.Ch00Screw),
      },
      wavelength: parseFloat(formData.wavelength),
      composition: formData.composition,
      sampleInfo: formData.sampleInfo
    };

    // 유효성 검사
    if (isNaN(materialConstants.latticeConstant) || materialConstants.latticeConstant <= 0) {
      alert('격자 상수는 0보다 큰 값이어야 합니다.');
      return;
    }

    if (isNaN(materialConstants.elasticConstants.C11) || 
        isNaN(materialConstants.elasticConstants.C12) || 
        isNaN(materialConstants.elasticConstants.C44)) {
      alert('모든 탄성 상수를 입력해주세요.');
      return;
    }

    if (Object.values(materialConstants.contrastFactors).some(v => isNaN(v) || v <= 0)) {
      alert('모든 전위 대비 인자/q 기준값은 0보다 큰 값이어야 합니다.');
      return;
    }

    if (isNaN(materialConstants.wavelength) || materialConstants.wavelength <= 0) {
      alert('X선 파장은 0보다 큰 값이어야 합니다.');
      return;
    }

    onSave(materialConstants);
    onClose();
  };

  if (!isOpen) return null;

  const qRange = getDislocationContrastPreset(formData.structure, formData.contrastFactors);

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 10000
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: 'white',
          padding: '30px',
          borderRadius: '8px',
          minWidth: '500px',
          maxWidth: '600px',
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 style={{ marginTop: 0, marginBottom: '20px' }}>
          재료 물성 상수 설정
        </h2>

        {/* 피크 정보로부터 자동 계산 */}
        {indexedPeaks && indexedPeaks.length >= 2 && (
          <div style={{ 
            marginBottom: '20px', 
            padding: '15px', 
            backgroundColor: 'var(--color-sub-2)', 
            borderRadius: '4px' 
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontWeight: '500' }}>XRD 피크 정보 기반 자동 계산</span>
              <button
                onClick={handleEstimateFromPeaks}
                className="btn-primary"
                style={{ padding: '6px 12px', fontSize: '12px' }}
              >
                격자 상수 자동 계산
              </button>
            </div>
            {estimatedLatticeConstant && (
              <div style={{ fontSize: '12px', color: 'var(--color-text-2)' }}>
                추정된 격자 상수: {estimatedLatticeConstant.toFixed(3)} Å
                {estimatedStructure && ` | 추정된 구조: ${estimatedStructure.toUpperCase()}`}
              </div>
            )}
            <div style={{ fontSize: '11px', color: 'var(--color-text-2)', marginTop: '5px' }}>
              사용 가능한 피크: {indexedPeaks.filter(p => p.millerIndices && p.millerIndices.length > 0).length}개
            </div>
          </div>
        )}

        {/* 시편 정보 */}
        <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: 'var(--color-sub-2)', borderRadius: '4px' }}>
          <h3 style={{ marginTop: 0, marginBottom: '15px', fontSize: '16px' }}>시편 정보</h3>
          
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
              조성 (Composition)
            </label>
            <textarea
              value={formData.composition}
              onChange={(e) => setFormData({ ...formData, composition: e.target.value })}
              placeholder="예: Fe-18Cr-8Ni (SUS304), Fe-0.45C (S45C) 등"
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '4px',
                border: '1px solid var(--color-monotone-2)',
                fontSize: '14px',
                minHeight: '60px',
                resize: 'vertical'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
              추가 시편 정보 (선택)
            </label>
            <textarea
              value={formData.sampleInfo}
              onChange={(e) => setFormData({ ...formData, sampleInfo: e.target.value })}
              placeholder="예: 냉간 압연 50%, 열처리 조건 등"
              style={{
                width: '100%',
                padding: '8px',
                borderRadius: '4px',
                border: '1px solid var(--color-monotone-2)',
                fontSize: '14px',
                minHeight: '60px',
                resize: 'vertical'
              }}
            />
          </div>
        </div>

        {/* 결정 구조 */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
            결정 구조 *
          </label>
          <select
            value={formData.structure}
            onChange={(e) => handleStructureChange(e.target.value)}
            style={{
              width: '100%',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid var(--color-monotone-2)',
              fontSize: '14px'
            }}
          >
            <option value="fcc">FCC (Face-Centered Cubic)</option>
            <option value="bcc">BCC (Body-Centered Cubic)</option>
          </select>
          <div style={{ marginTop: '5px', fontSize: '12px', color: 'var(--color-text-2)' }}>
            전위 특성 q 기준: {qRange.min.toFixed(2)} ~ {qRange.max.toFixed(2)}
            (나선형: {qRange.screw.toFixed(2)}, 칼날형: {qRange.edge.toFixed(2)})
          </div>
        </div>

        {/* 전위 대비 인자 */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
            mWH 전위 대비 인자/q 기준 *
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>q edge</label>
              <input
                type="number"
                step="0.001"
                value={formData.contrastFactors.qEdge}
                onChange={(e) => setFormData({
                  ...formData,
                  contrastFactors: { ...formData.contrastFactors, qEdge: e.target.value }
                })}
                style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--color-monotone-2)', fontSize: '14px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>q screw</label>
              <input
                type="number"
                step="0.001"
                value={formData.contrastFactors.qScrew}
                onChange={(e) => setFormData({
                  ...formData,
                  contrastFactors: { ...formData.contrastFactors, qScrew: e.target.value }
                })}
                style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--color-monotone-2)', fontSize: '14px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Cₕ₀₀ edge</label>
              <input
                type="number"
                step="0.001"
                value={formData.contrastFactors.Ch00Edge}
                onChange={(e) => setFormData({
                  ...formData,
                  contrastFactors: { ...formData.contrastFactors, Ch00Edge: e.target.value }
                })}
                style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--color-monotone-2)', fontSize: '14px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>Cₕ₀₀ screw</label>
              <input
                type="number"
                step="0.001"
                value={formData.contrastFactors.Ch00Screw}
                onChange={(e) => setFormData({
                  ...formData,
                  contrastFactors: { ...formData.contrastFactors, Ch00Screw: e.target.value }
                })}
                style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--color-monotone-2)', fontSize: '14px' }}
              />
            </div>
          </div>
          <div style={{ marginTop: '5px', fontSize: '12px', color: 'var(--color-text-2)' }}>
            {qRange.source}. q로 screw fraction S=(q-qEdge)/(qScrew-qEdge)를 추정합니다.
          </div>
        </div>

        {/* 격자 상수 */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontWeight: '500' }}>
              격자 상수 a (Å) *
            </label>
            {estimatedLatticeConstant && (
              <span style={{ fontSize: '12px', color: 'var(--color-primary)', fontWeight: 'bold' }}>
                추정값: {estimatedLatticeConstant.toFixed(3)} Å
              </span>
            )}
          </div>
          <input
            type="number"
            step="0.001"
            value={formData.latticeConstant}
            onChange={(e) => setFormData({ ...formData, latticeConstant: e.target.value })}
            style={{
              width: '100%',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid var(--color-monotone-2)',
              fontSize: '14px'
            }}
            placeholder={estimatedLatticeConstant ? `예: ${estimatedLatticeConstant.toFixed(3)}` : "예: 3.615"}
          />
          <div style={{ marginTop: '5px', fontSize: '12px', color: 'var(--color-text-2)' }}>
            {estimatedLatticeConstant 
              ? `피크 정보로부터 추정: ${estimatedLatticeConstant.toFixed(3)} Å`
              : formData.structure === 'fcc' ? 'SUS304 기본값: 3.615 Å' : 'α-Fe 기본값: 2.866 Å'}
          </div>
        </div>

        {/* 탄성 상수 */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <label style={{ fontWeight: '500' }}>탄성 상수 (GPa) *</label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '14px', fontWeight: 'normal' }}>
              <input
                type="checkbox"
                checked={useDefaults}
                onChange={(e) => handleUseDefaultsToggle(e.target.checked)}
                style={{ marginRight: '5px' }}
              />
              기본값 사용
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                C₁₁
              </label>
              <input
                type="number"
                step="0.1"
                value={formData.elasticConstants.C11}
                onChange={(e) => setFormData({
                  ...formData,
                  elasticConstants: { ...formData.elasticConstants, C11: e.target.value }
                })}
                disabled={useDefaults}
                style={{
                  width: '100%',
                  padding: '6px',
                  borderRadius: '4px',
                  border: '1px solid var(--color-monotone-2)',
                  fontSize: '14px',
                  opacity: useDefaults ? 0.6 : 1
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                C₁₂
              </label>
              <input
                type="number"
                step="0.1"
                value={formData.elasticConstants.C12}
                onChange={(e) => setFormData({
                  ...formData,
                  elasticConstants: { ...formData.elasticConstants, C12: e.target.value }
                })}
                disabled={useDefaults}
                style={{
                  width: '100%',
                  padding: '6px',
                  borderRadius: '4px',
                  border: '1px solid var(--color-monotone-2)',
                  fontSize: '14px',
                  opacity: useDefaults ? 0.6 : 1
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px' }}>
                C₄₄
              </label>
              <input
                type="number"
                step="0.1"
                value={formData.elasticConstants.C44}
                onChange={(e) => setFormData({
                  ...formData,
                  elasticConstants: { ...formData.elasticConstants, C44: e.target.value }
                })}
                disabled={useDefaults}
                style={{
                  width: '100%',
                  padding: '6px',
                  borderRadius: '4px',
                  border: '1px solid var(--color-monotone-2)',
                  fontSize: '14px',
                  opacity: useDefaults ? 0.6 : 1
                }}
              />
            </div>
          </div>
          <div style={{ marginTop: '5px', fontSize: '12px', color: 'var(--color-text-2)' }}>
            {formData.structure === 'fcc' 
              ? 'SUS304 기본값: C₁₁=204.6, C₁₂=137.7, C₄₄=126.2 GPa'
              : 'α-Fe 기본값: C₁₁=231.4, C₁₂=134.7, C₄₄=116.4 GPa'}
          </div>
        </div>

        {/* X선 파장 */}
        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
            X선 파장 (nm) *
          </label>
          <input
            type="number"
            step="0.00001"
            value={formData.wavelength}
            onChange={(e) => setFormData({ ...formData, wavelength: e.target.value })}
            style={{
              width: '100%',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid var(--color-monotone-2)',
              fontSize: '14px'
            }}
            placeholder="예: 0.15405"
          />
          <div style={{ marginTop: '5px', fontSize: '12px', color: 'var(--color-text-2)' }}>
            Cu Kα 기본값: 0.15405 nm (1.5405 Å)
          </div>
        </div>

        {/* 버튼 */}
        <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '30px' }}>
          <button
            onClick={onClose}
            style={{
              padding: '10px 20px',
              background: 'var(--color-monotone-2)',
              color: 'var(--color-text-1)',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            취소
          </button>
          <button
            onClick={handleSave}
            className="btn-primary"
            style={{
              padding: '10px 20px',
              fontSize: '14px'
            }}
          >
            저장
          </button>
        </div>
      </div>
    </div>
  );
};

export default MaterialConstantsModal;

