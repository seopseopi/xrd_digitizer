import React from 'react';
import { useLocation } from 'react-router-dom';
import { useToolbarContent } from '../context/ToolbarContentContext';

const Toolbar = () => {
  const location = useLocation();
  const { toolbarContent, toolbarFooterContent } = useToolbarContent();

  // 각 페이지별 툴 정의
  const getPageTools = () => {
    const path = location.pathname;

    if (path.startsWith('/research')) {
      return [];
    }

    if (path.startsWith('/experiment')) {
      return [];
    }

    if (path.startsWith('/data')) {
      return [];
    }

    if (path.startsWith('/tools')) {
      return [];
    }

    if (path.startsWith('/admin')) {
      return [];
    }

    if (path === '/profile') {
      return [];
    }

    return [];
  };

  const tools = getPageTools();

  return (
    <div className="box-col group-align-std" id="toolbar">
      <div className="toolbar-inner">
        <div className="toolbar-header-strip">
          <h4 className="toolbar-title">도구</h4>
        </div>

        <div className="toolbar-scroll-area">
          {toolbarContent ? (
            <div className="toolbar-content">{toolbarContent}</div>
          ) : tools.length > 0 ? (
            tools.map((tool) => (
              <div
                key={tool.id}
                className="side-menu-item"
                onClick={tool.onClick}
                style={{
                  cursor: 'pointer',
                  justifyContent: 'flex-start',
                }}
              >
                {tool.icon && (
                  <span className="material-symbols-rounded">{tool.icon}</span>
                )}
                <span style={{ marginLeft: '5px' }}>{tool.label}</span>
              </div>
            ))
          ) : (
            <div className="toolbar-empty-hint">사용 가능한 도구가 없습니다</div>
          )}
        </div>

        {toolbarFooterContent ? (
          <div className="toolbar-footer">{toolbarFooterContent}</div>
        ) : null}
      </div>
    </div>
  );
};

export default Toolbar;
