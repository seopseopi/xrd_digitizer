import React, { createContext, useState, useContext } from 'react';

const ToolbarContentContext = createContext();

export const useToolbarContent = () => {
  const context = useContext(ToolbarContentContext);
  if (!context) {
    throw new Error('useToolbarContent must be used within a ToolbarContentProvider');
  }
  return context;
};

export const ToolbarContentProvider = ({ children }) => {
  const [toolbarContent, setToolbarContent] = useState(null);
  const [toolbarFooterContent, setToolbarFooterContent] = useState(null);
  return (
    <ToolbarContentContext.Provider
      value={{
        toolbarContent,
        setToolbarContent,
        toolbarFooterContent,
        setToolbarFooterContent,
      }}
    >
      {children}
    </ToolbarContentContext.Provider>
  );
};
