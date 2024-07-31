import React from 'react';
import { FormControl, Select, MenuItem, SelectChangeEvent, InputLabel } from '@mui/material';
import styled from '@emotion/styled';

const StyledFormControl = styled(FormControl)`
margin: 8px;
min-width: 120px;
width: 100%;
`;

const StyledInputLabel = styled(InputLabel)`
  margin-bottom: 5px;  /* 或者使用 padding: 5px; */
`;

interface CustomFormControlProps {
  label: string;
  value: string;
  onChange: (event: SelectChangeEvent) => void;
  options: Array<{ value: string; label: string }>;
}

const CustomFormControl: React.FC<CustomFormControlProps> = ({ label, value, onChange, options }) => {

  return (
    <StyledFormControl size='small'>
      <StyledInputLabel>{label}</StyledInputLabel>
      <Select 
        label={label} 
        value={value} 
        onChange={onChange}
        labelId={`select-${label}-label`}
      >
        {options.map((option) => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </Select>
    </StyledFormControl>
  );
};

export default CustomFormControl;