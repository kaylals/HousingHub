import React from 'react';
import { FormControl, Select, MenuItem, SelectChangeEvent, InputLabel } from '@mui/material';
import styled from '@emotion/styled';

const StyledFormControl = styled(FormControl)`
margin: 8px;
min-width: 120px;
width: 100%;
`;

const StyledInputLabel = styled(InputLabel)`
  margin-bottom: 5px;
`;

interface CustomDropDownProps {
  label: string;
  value: any;
  onChange: (event: SelectChangeEvent) => void;
  options: Array<{ value: string; label: string } | { value: number; label: number }>;
}

const CustomDropDown: React.FC<CustomDropDownProps> = ({ label, value, onChange, options }) => {

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

export default CustomDropDown;